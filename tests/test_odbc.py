"""Tests for NetSuite SuiteAnalytics Connect ODBC support.

The core of `NetSuiteODBC` is connection-string assembly; the runtime
calls (`connect`, `execute`, `query`) just delegate to pyodbc. We test
the assembly directly and use mocking for the delegation paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from netsuite import NetSuiteODBC
from netsuite.config import Config, UsernamePasswordAuth
from netsuite.odbc import PYODBC_INSTALLED

# ---------------------------------------------------------------------------
# Existing config-level tests (kept from before)
# ---------------------------------------------------------------------------


def test_tba(dummy_config):
    assert dummy_config.is_token_auth


def test_sandbox_account(dummy_config):
    assert dummy_config.is_sandbox
    assert dummy_config.account_number == "123456"
    assert dummy_config.account_slugified == "123456-sb1"


def test_production_account_extraction(dummy_config_with_production_account):
    assert "_SB" not in dummy_config_with_production_account.account
    assert dummy_config_with_production_account.account_number == "123456"
    assert dummy_config_with_production_account.is_sandbox is False
    assert dummy_config_with_production_account.account_slugified == "123456"


def test_username_auth(dummy_username_password_config):
    config = dummy_username_password_config

    assert config.auth.username == "username"
    assert not config.is_token_auth
    assert config.is_password_auth


# ---------------------------------------------------------------------------
# Config additions
# ---------------------------------------------------------------------------


def test_username_auth_role_optional():
    auth = UsernamePasswordAuth(username="u", password="p")
    assert auth.role is None


def test_username_auth_role_can_be_int_or_str():
    assert UsernamePasswordAuth(username="u", password="p", role=3).role == 3
    assert (
        UsernamePasswordAuth(username="u", password="p", role="admin").role == "admin"
    )


def test_default_odbc_driver():
    config = Config(account="123", auth={"username": "u", "password": "p"})
    assert config.odbc_driver == "{NetSuite Drivers 64bit}"


def test_default_odbc_data_source():
    config = Config(account="123", auth={"username": "u", "password": "p"})
    assert config.odbc_data_source == "NetSuite.com"


def test_odbc_driver_overridable():
    config = Config(
        account="123",
        auth={"username": "u", "password": "p"},
        odbc_driver="MyCustomDriver",
    )
    assert config.odbc_driver == "MyCustomDriver"


def test_odbc_data_source_rejects_invalid_value():
    """`odbc_data_source` is a Literal — Pydantic should reject other values."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config(
            account="123",
            auth={"username": "u", "password": "p"},
            odbc_data_source="bogus",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# is_password_auth dispatch
# ---------------------------------------------------------------------------


def test_token_config_is_not_password_auth(dummy_config):
    assert not dummy_config.is_password_auth


def test_password_config_is_not_token_auth(dummy_username_password_config):
    assert not dummy_username_password_config.is_token_auth
    assert dummy_username_password_config.is_password_auth


# ---------------------------------------------------------------------------
# NetSuiteODBC initialization
# ---------------------------------------------------------------------------


pytestmark_pyodbc = pytest.mark.skipif(not PYODBC_INSTALLED, reason="Requires pyodbc")


def test_odbc_rejects_token_auth(dummy_config):
    """Token auth makes no sense for ODBC — surface a clear error."""
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    with pytest.raises(RuntimeError, match="UsernamePasswordAuth"):
        NetSuiteODBC(dummy_config)


def test_odbc_init_accepts_username_password(dummy_username_password_config):
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    odbc = NetSuiteODBC(dummy_username_password_config)
    assert odbc._config is dummy_username_password_config


def test_odbc_hostname_uses_account_subdomain(dummy_username_password_config):
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    odbc = NetSuiteODBC(dummy_username_password_config)
    assert odbc.hostname == "123456-sb1.connect.api.netsuite.com"


# ---------------------------------------------------------------------------
# Connection-string assembly
# ---------------------------------------------------------------------------


def _odbc_with_role(role):
    config = Config(
        account="123_SB1",
        auth={"username": "u", "password": "p", "role": role},
    )
    if not PYODBC_INSTALLED:
        return None
    return NetSuiteODBC(config)


def test_connection_string_includes_all_fields(dummy_username_password_config):
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    odbc = NetSuiteODBC(dummy_username_password_config)
    cs = odbc.connection_string
    assert "DRIVER={NetSuite Drivers 64bit}" in cs
    assert "Host=123456-sb1.connect.api.netsuite.com" in cs
    assert "Port=1708" in cs
    assert "UID=username" in cs
    assert "PWD=password" in cs
    assert "ServerDataSource=NetSuite.com" in cs
    assert "AccountID=123456_SB1" in cs


def test_connection_string_includes_role_when_set():
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    odbc = _odbc_with_role(role=3)
    assert "RoleID=3" in odbc.connection_string


def test_connection_string_omits_role_when_unset(dummy_username_password_config):
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    odbc = NetSuiteODBC(dummy_username_password_config)
    assert "RoleID=" not in odbc.connection_string


def test_connection_string_uses_custom_driver():
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    config = Config(
        account="123",
        auth={"username": "u", "password": "p"},
        odbc_driver="MyDriver",
    )
    odbc = NetSuiteODBC(config)
    assert "DRIVER=MyDriver" in odbc.connection_string


# ---------------------------------------------------------------------------
# query() / execute() delegation. We mock pyodbc.connect so no driver is
# required to exercise these.
# ---------------------------------------------------------------------------


def test_query_returns_list_of_dicts(dummy_username_password_config):
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    odbc = NetSuiteODBC(dummy_username_password_config)

    mock_cursor = MagicMock()
    mock_cursor.description = [("id",), ("name",)]
    mock_cursor.fetchall.return_value = [(1, "alice"), (2, "bob")]
    mock_cursor.execute.return_value = mock_cursor

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = False
    mock_conn.cursor.return_value = mock_cursor

    with patch("netsuite.odbc._pyodbc.connect", return_value=mock_conn) as mock_connect:
        rows = odbc.query("SELECT id, name FROM customer")

    mock_connect.assert_called_once_with(odbc.connection_string)
    mock_cursor.execute.assert_called_once_with("SELECT id, name FROM customer")
    assert rows == [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]


def test_execute_passes_sql_through(dummy_username_password_config):
    if not PYODBC_INSTALLED:
        pytest.skip("Requires pyodbc")
    odbc = NetSuiteODBC(dummy_username_password_config)

    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = False
    mock_conn.cursor.return_value = mock_cursor

    with patch("netsuite.odbc._pyodbc.connect", return_value=mock_conn):
        odbc.execute("UPDATE customer SET active=1")

    mock_cursor.execute.assert_called_once_with("UPDATE customer SET active=1")


# ---------------------------------------------------------------------------
# Missing-pyodbc behavior
# ---------------------------------------------------------------------------


def test_init_raises_when_pyodbc_not_installed(dummy_username_password_config):
    """If `pyodbc` is unavailable, instantiating NetSuiteODBC must fail
    with a message pointing the user at the `odbc` extra."""
    with patch("netsuite.odbc.PYODBC_INSTALLED", False):
        with pytest.raises(RuntimeError, match="odbc"):
            NetSuiteODBC(dummy_username_password_config)

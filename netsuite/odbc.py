"""SuiteAnalytics Connect ODBC client.

NetSuite exposes a relational view of an account's data through
SuiteAnalytics Connect, which speaks ODBC. This module wraps `pyodbc`
to make connecting to it from Python a one-liner once the platform
ODBC driver is installed.

Imports the implementation from upstream PR jacobsvante/netsuite#122
(tonymorello), with adjustments for Python 3.9 compatibility and to
keep the existing UsernamePasswordAuth deprecation language.
"""

import typing as t

from .config import Config, UsernamePasswordAuth

# `pyodbc` is an optional dependency installed via the `odbc` extra. We
# lazy-import it so the rest of the package still imports cleanly when
# the extra isn't installed; only callers that actually instantiate
# `NetSuiteODBC` will need it.
try:
    import pyodbc as _pyodbc

    PYODBC_INSTALLED = True
except ImportError:
    _pyodbc = None  # type: ignore[assignment]
    PYODBC_INSTALLED = False

__all__ = ("NetSuiteODBC", "PYODBC_INSTALLED")


class NetSuiteODBC:
    """A connection helper for NetSuite's SuiteAnalytics Connect ODBC service.

    Uses `Config.auth` (must be `UsernamePasswordAuth`) and
    `Config.odbc_data_source` / `Config.odbc_driver` to assemble a
    connection string, then exposes `connect()`, `execute()`, and
    `query()` for common operations.
    """

    def __init__(self, config: Config):
        if not PYODBC_INSTALLED:
            raise RuntimeError(
                "pyodbc is required for ODBC connections. "
                "Install with `pip install 'netsuite[odbc] @ "
                "git+https://github.com/vlouvet/netsuite.git'`."
            )
        if not config.is_password_auth:
            raise RuntimeError(
                "ODBC connections require UsernamePasswordAuth. "
                "TokenAuth and OAuth 2.0 are not supported by "
                "SuiteAnalytics Connect."
            )
        self._config = config

    @property
    def hostname(self) -> str:
        return f"{self._config.account_slugified}.connect.api.netsuite.com"

    @property
    def connection_string(self) -> str:
        auth = self._config.auth
        # Sanity-checked in __init__ but mypy needs the narrow.
        assert isinstance(auth, UsernamePasswordAuth)
        role_part = f"RoleID={auth.role}" if auth.role is not None else ""
        return ";".join(
            part
            for part in (
                f"DRIVER={self._config.odbc_driver}",
                f"Host={self.hostname}",
                "Port=1708",
                "Encrypted=1",
                "AllowSinglePacketLogout=1",
                "Truststore=system",
                f"ServerDataSource={self._config.odbc_data_source}",
                f"UID={auth.username}",
                f"PWD={auth.password}",
                (
                    f"CustomProperties=AccountID={self._config.account};" f"{role_part}"
                ).rstrip(";"),
            )
            if part
        )

    def connect(self):
        """Open a new pyodbc connection. Caller is responsible for closing."""
        return _pyodbc.connect(self.connection_string)

    def execute(self, sql: str):
        """Execute a SQL statement and return the pyodbc cursor.

        Note: the connection is closed when this returns. For workloads
        where you need to iterate cursor rows lazily, use `connect()`
        directly and manage the connection yourself.
        """
        with self.connect() as connection:
            cursor = connection.cursor()
            return cursor.execute(sql)

    def query(self, sql: str) -> t.List[t.Dict[str, t.Any]]:
        """Execute a SQL statement and return all rows as a list of dicts."""
        with self.connect() as connection:
            cursor = connection.cursor()
            response = cursor.execute(sql)
            headers = [col[0] for col in response.description]
            return [dict(zip(headers, row)) for row in response.fetchall()]

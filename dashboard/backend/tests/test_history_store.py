from dashboard.backend.app.history_store import MySQLSimulationHistoryStore


class Cursor:
    def __init__(self):
        self.statements = []
        self.lastrowid = 41

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        self.statements.append((sql, params))

    def executemany(self, sql, params):
        self.statements.append((sql, params))


class Connection:
    def __init__(self):
        self.cursor_value = Cursor()
        self.committed = False

    def cursor(self):
        return self.cursor_value

    def commit(self):
        self.committed = True

    def rollback(self):
        raise AssertionError("unexpected rollback")

    def close(self):
        pass


def test_history_write_is_additive_and_never_changes_financial_tables(monkeypatch):
    connection = Connection()
    monkeypatch.setattr(MySQLSimulationHistoryStore, "_connect", lambda self: connection)
    store = MySQLSimulationHistoryStore.__new__(MySQLSimulationHistoryStore)
    result = {"transactionID": 22, "processID": 1, "burstTime": 2,
              "completionTime": 2, "waitingTime": 0,
              "turnaroundTime": 2, "responseTime": 0}
    assert store.create(
        algorithm="fcfs", parameters={}, workload=[{"transactionID": 22, "burstTime": 2}],
        summary={"transactionCount": 1, "makespan": 2, "averageWaitingTime": 0,
                 "averageTurnaroundTime": 2, "averageResponseTime": 0,
                 "cpuUtilization": 100, "throughput": 0.5},
        results=[result],
    ) == 41
    sql = [statement.lower() for statement, _params in connection.cursor_value.statements]
    assert len(sql) == 2
    assert all("insert into simulationhistory" in statement for statement in sql)
    assert all("accounts" not in statement and "transactions" not in statement for statement in sql)
    assert connection.committed

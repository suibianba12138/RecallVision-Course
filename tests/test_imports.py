"""Import smoke tests for the P0 package surface."""


def test_recallvision_schema_modules_import() -> None:
    import recallvision
    from recallvision.events import schema as events_schema
    from recallvision.memory import schema as memory_schema
    from recallvision.query import schema as query_schema

    assert recallvision is not None
    assert memory_schema is not None
    assert events_schema is not None
    assert query_schema is not None

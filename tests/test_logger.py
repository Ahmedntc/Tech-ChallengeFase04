import json
import logging

from credit_monitoring.monitoring.logger import configure_logging, log_event


def _events(settings) -> list[dict]:
    with open(settings["paths"]["logs_file"], encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_events_are_written_as_json_lines_with_run_id_and_fields(settings):
    run_id = configure_logging(settings)
    logger = logging.getLogger("credit_monitoring.teste")
    log_event(logger, logging.WARNING, "alerta", "Drift em age", month="M4", value=0.136)

    (event,) = _events(settings)
    assert event["run_id"] == run_id
    assert (event["level"], event["event"], event["message"]) == (
        "WARNING",
        "alerta",
        "Drift em age",
    )
    assert (event["month"], event["value"]) == ("M4", 0.136)
    assert event["logger"] == "credit_monitoring.teste"


def test_plain_log_calls_get_a_default_event(settings):
    configure_logging(settings)
    logging.getLogger("credit_monitoring.teste").info("Lote %s aprovado", "M1")
    (event,) = _events(settings)
    assert (event["event"], event["message"]) == ("log", "Lote M1 aprovado")


def test_third_party_info_is_dropped_but_warnings_are_kept(settings):
    configure_logging(settings)
    logging.getLogger("biblioteca.externa").info("ruído")
    logging.getLogger("biblioteca.externa").warning("algo importante")
    assert [e["message"] for e in _events(settings)] == ["algo importante"]


def test_reconfiguring_starts_a_new_run_without_duplicating_lines(settings):
    first = configure_logging(settings)
    second = configure_logging(settings)
    logging.getLogger("credit_monitoring.teste").info("uma vez")

    events = _events(settings)
    assert first != second
    assert len(events) == 1
    assert events[0]["run_id"] == second


def test_exceptions_carry_the_traceback(settings):
    configure_logging(settings)
    try:
        raise RuntimeError("quebrou")
    except RuntimeError:
        logging.getLogger("credit_monitoring.teste").exception("Falha na tarefa")
    (event,) = _events(settings)
    assert event["level"] == "ERROR"
    assert "RuntimeError: quebrou" in event["exception"]

"""SFR-003: every command-line option is described in the help output."""

import argparse

import pytest

from fabricpc.bench.__main__ import _parser as main_parser
from fabricpc.bench.cli.explicit import list_parser, run_parser
from fabricpc.bench.cli.queue import _parser as queue_parser

PARSERS = {
    "main": main_parser,
    "queue": queue_parser,
    "run": run_parser,
    "list": list_parser,
}


@pytest.mark.parametrize("name", PARSERS)
def test_every_option_has_help_text(name):
    parser = PARSERS[name]()
    missing = [
        "/".join(a.option_strings) or a.dest
        for a in parser._actions
        if not isinstance(a, argparse._HelpAction)
        and (not a.help or a.help == argparse.SUPPRESS)
    ]
    assert missing == [], f"{name}: options with no help text: {missing}"


@pytest.mark.parametrize("name", PARSERS)
def test_every_option_shows_up_in_help(name):
    parser = PARSERS[name]()
    text = parser.format_help()
    for action in parser._actions:
        for flag in action.option_strings:
            assert flag in text, f"{name}: {flag} is missing from --help"

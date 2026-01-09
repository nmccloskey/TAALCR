#!/usr/bin/env python3
"""
TAAALCR Command-Line Interface
----------------------------
Entry point for the Toolkit for Aggregate Analysis and Alignment of Language in Conversation, for Research (TAAALCR).
Delegates parser construction to main.build_arg_parser().
"""

from .main import main as main_core
from taaalcr.utils.auxiliary import build_arg_parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    main_core(args)


if __name__ == "__main__":
    main()

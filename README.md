# UMW-GBUS-562
Repository for GBUS 562 (Al Language Models)

## About

This repository contains materials created for GBUS 562: AI Language Models at the University of Mary Washington (UMW). The content is organized into weekly folders, one for each week of the course.

## Weekly Work

1. [week-01](week-01/)
   - [Connect 4](week-01/connect4.html): a single-page browser game built with HTML and JavaScript. Play against a computer opponent that looks several moves ahead (Easy, Medium, or Hard), or switch to two-player mode and take turns on the same device.
2. [week-02](week-02/)
   - [Prompt Lab](week-02/prompt-lab.html): a single-file web app for testing prompts against Claude, built with Claude Code.
3. [week-03](week-03/)
   - Week 3: a RAG tool that answers questions from a folder of documents (Copperline Manufacturing's finance documents by default, or Larkfield Credit Union's member documents with `--docs week-03/larkfield-docs`), built with Claude Code.
4. [week-04](week-04/)
   - Week 4: an AI agent that prepares replies to HR emails using tools from an MCP server, built with Claude Code. It can handle one email file, or (`--inbox`) every unread email in a Gmail sandbox, saving each reply as a Gmail draft in the same thread. It never sends email. `load_inbox.py` fills the sandbox inbox with the sample emails.

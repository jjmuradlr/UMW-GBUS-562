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
   - Week 4: an AI agent that prepares replies to HR emails using tools from an MCP server, built with Claude Code. It can handle one email file, or (`--inbox`) every unread email in a Gmail sandbox, saving each reply as a Gmail draft in the same thread. With `--watch` it checks the inbox every 30 seconds, answers only allowlisted senders, and sends up to 5 replies that don't need human review (the rest stay as drafts); Claude itself never gets a send tool. `--org` points it at any organization folder (policy documents, employee file, sample emails, system prompt); it ships with Hartwell Foods (default) and Ridgeline Hotels (`--org ridgeline-hotels`), and works out the employee file's columns from its header row. `load_inbox.py` fills the sandbox inbox with the sample emails.
5. [week-05](week-05/)
   - Week 5: a team of AI agents that prepares a bicycle maker's response to supplier emails, built with Claude Code.
   - [specialists.py](week-05/specialists.py): the four specialists the coordinators share: a supply analyst (an agent using the MCP server's tools, at most 10 tool calls, returns a JSON brief), a supplier writer, a customer writer, and a reviewer. Every draft is reviewed and revised at most twice, checked in code for customer or supplier names, and saved in `week-05/drafts`. A draft is marked for a person if it fails a check or its order has a late penalty.
   - [pipeline.py](week-05/pipeline.py): runs the specialists on one supplier email (`--email`) in a fixed order: the analyst, a customer notice for each order at risk, then a reply to the supplier. It ends with a summary of drafts saved, drafts marked for a person, decisions for a person, and total tokens. `--save FILE` also writes everything printed to a UTF-8 text file.
   - [orchestrator.py](week-05/orchestrator.py): the same job, but Claude decides which specialists to use. An orchestrator agent gets four tools (`run_supply_analyst`, `draft_supplier_reply`, `draft_customer_notice`, `finish`), stops after 10 tool calls, prints each choice it makes, and ends with the pipeline's summary plus its own summary and whether a person must make a decision. Same `--email` and `--save` options.

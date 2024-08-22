## Definition of Data Agents

Data Agents are Large Language Model (LLM) powered knowledge workers capable of performing tasks over data with both "read" and "write" functions.

### Capabilities

- Automated search and retrieval over various data types.
- Calling external service APIs and processing or caching responses.
- Storing conversation history.
- Performing complex data tasks by combining various functions.

## Components for Building Data Agents

General Agent/Tool Abstractions: These abstractions help build agent loops and interact with tools via structured API definitions.

### LlamaHub Tool Repository

A collection of tools (e.g., Google Calendar, Notion, SQL, OpenAPI) that can be connected and used by agents, open to community contributions.

## Enhanced Query Engines

Previous focus was on improving query engines for diverse and complex queries, including chain-of-thought reasoning and query decomposition.

## Opportunities for LLMs

- General Reasoning and Interaction: Enabling LLMs to interact with various tools for both understanding and modifying data states.
- Beyond Search and Retrieval: Expanding functionalities beyond static knowledge source searches.

## Existing Approaches and New Opportunities

- Inspiration from Existing Solutions: Building upon existing LLM-powered agent demonstrations to create a more structured and comprehensive system for data agents.

## Core Components of Data Agents

### Reasoning Loop

A process where the agent decides which tools to use, in what sequence, and with what parameters. This loop can be simple or complex.

### Tool Abstractions

APIs and Tools that return information or perform state-modifying actions.

### Agent Abstractions and Reasoning Loops

- OpenAI Function Agent: Uses the OpenAI Function API for tool decision logic within a loop.
- ReAct Agent: Uses general text completion endpoints with prompt-based reasoning logic, inspired by the ReAct paper.

## Tool Abstractions

### Base Tool Abstraction

Defines generic interfaces for tools, including metadata and response formats.

### Function Tool

Converts user-defined functions into tools with auto-inferred schemas.

### QueryEngineTool

Wraps existing query engines into tools for seamless integration into agent settings.



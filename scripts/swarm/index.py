from swarm import Swarm, Agent

from repl import run_demo_loop

client = Swarm()

orchestrator = Agent(
    name="Orchestrator",
    instructions='''
You are an orchestrator. Your job is to process user requests. To process a request, you need to follow this flow:
1. Think about the steps necessary to resolve the user request.
2. Think what agents are needed to resolve the user request.
3. Transfer the user request to the appropriate agent.
4. Wait for the agent to respond.
5. Based on agent's response, decide the next steps.
6. When finished with the task write "Conversation complete." to end the conversation.''',
)


def transfer_to_SQL_agent():
    """Transfer to sql agent when you need to generate a SQL query."""
    return sql_agent


def transfer_to_orchestrator():
    """Transfer back to orchestrator when you have finished your task."""
    return orchestrator


def get_sql_function(query):
    """provides a working SQL query for the given user query"""
    def test_sql(query):
        """tests if SQL query is working"""
        return "[mock]: SQL query is working"
    stop = False
    client = Swarm()
    sql_agent = Agent(
        name="SQL agent",
        instructions="You are an SQL agent. Your job is to generate SQL queries. You should validate that your query is working before sending it off the the user.",
        functions=[test_sql]
    )

    messages = [{"role": "user", "content": query}]
    response = client.run(agent=sql_agent, messages=messages,
                          context_variables={"stop": stop})
    return response.messages[-1]["content"]

# role -> agents -> functions/actions

# agent
# prompt
# context = messages
# actions
# hand_over


messages = [{"role": "user",
             "content": "I want to know a query for getting all users of age higher than 30."}]
# response = client.run(agent=orchestrator, messages=messages)

# for message in response.messages:
# print(f"{message['role']}: {message['content']}")

run_demo_loop(orchestrator, None, True, True)

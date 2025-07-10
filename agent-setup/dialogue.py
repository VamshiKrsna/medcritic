import asyncio, os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

llm = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

doctor = AssistantAgent(
    name="Doctor",
    description="A physician assessing patient's symptoms by asking questions.",
    model_client=llm,
    system_message=(
        "You are a compassionate doctor. Ask clear and concise questions "
        "to understand the patient's symptoms, history, and exam findings."
    ),
)

patient = AssistantAgent(
    name="Patient",
    description="A patient who only knows their symptoms and answers honestly.",
    model_client=llm,
    system_message=(
        "You are a patient. Answer the doctor's questions based only on your symptoms; "
        "do not speculate about diagnosis."
    ),
)

critic = AssistantAgent(
    name="Critic",
    description="Observes the conversation and produces a SOAP clinical summary.",
    model_client=llm,
    system_message=(
        "You are a medical Critic. After conversation ends, produce a structured SOAP note:\n"
        "- Subjective: patient complaints/history\n"
        "- Objective: observed signs/data\n"
        "- Assessment: likely diagnosis\n"
        "- Plan: recommended next steps/tests/treatment\n"
    ),
)

async def run_convo():
    team = RoundRobinGroupChat(
        participants=[doctor, patient],
        group_chat_manager_class = critic,
        termination_condition=MaxMessageTermination(max_messages=10),
        max_turns=10
    )
    result = await team.run(task="Begin the consultation.")
    transcript = result.messages  
    full_transcript = "\n".join(
        f"{msg.source}: {msg.content}" for msg in transcript if hasattr(msg, "source")
    )

    print("\n=== Conversation Transcript ===")
    print(full_transcript)
    print("\n=== Critic SOAP Summary ===")
    print(result.get_manager_response().content)

if __name__ == "__main__":
    asyncio.run(run_convo())
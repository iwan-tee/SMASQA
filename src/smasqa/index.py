from smasqa.agents.orchestrator import Orchestrator

task = "What’s the typical amount passengers usually pay?"
db_name = "test_ave.db"
db_description = "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked,AgeBand"
options = ["Answer 1: 14.45", "Answer 2: 1412837", "Answer 3: 726"]
orchestrator = Orchestrator(
    task=f"""Your task is: {task}
    Additional info that you'll need: 
    Database header row(description): {db_description}
    Database name: {db_name}""",
    options=options
)

print("Getting SMASQA... ")
result = orchestrator.run()
print(f'Result: {result}')
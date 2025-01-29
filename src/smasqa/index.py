from smasqa.agents.orchestrator import Orchestrator


task = "What’s the typical amount passengers usually pay?"
db_name = "test_ave.db"
db_description = "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked,AgeBand"

orchestrator = Orchestrator(
    task=f"""Your task is: {task}
    Additional info that you'll need: 
    Database header row(description): {db_description}
    Database name: {db_name}""",

)

print("Getting SMASQA... ")
orchestrator.run()

# SMASQA

This is a repository for our SQL Multi-Agent System for Question-Answering as part of the "Table Representation Learning" seminar.

Setup Instructions
1. **Clone the Repository**
```bash
git clone git@github.com:iwan-tee/SMASQA.git
cd your_project
```
2. **Install Poetry**
If you don't have Poetry installed:
For Unix/macOS:

```bash
Copy code
curl -sSL https://install.python-poetry.org | python3 -
```
For Windows (PowerShell):

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```
After installation, ensure that Poetry is added to your system's PATH.

4. **Enter virtual env**
Start poetry environment:
```bash
poetry shell
```
*Note:* to exit virtual env use ```exit```

5. **Install Dependencies**
Install the project dependencies using Poetry:

```bash
poetry install
```

6. **Set Up Environment Variables**
Copy the example .env file and add your API keys:

```bash
cp .env.example .env
```
Edit the .env file and replace placeholders with your actual API keys.

7. **Run the Hello Script Using Poe the Poet**

Since poethepoet is configured in the project, you can run the hello script using:
```bash
poe hello
```
This command runs the hello task defined in your pyproject.toml using poethepoet.
The script scripts/hello.py will execute and print out all collected environment variables.

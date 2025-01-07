from qs import qs

questions = [[item['id'], item['question']] for item in qs]

# Print the result
print(questions[:30])

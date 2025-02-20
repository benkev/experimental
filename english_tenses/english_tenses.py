 
# Define tense examples
tenses = [
    ("Simple", "Progressive", "Perfect", "Perfect Progressive"),
    ("Present", "Past", "Future", "Future in the Past"),
]

examples = {
    ("Present", "Simple"): [
        "I play tennis every Sunday.",
        "Do you play tennis every Sunday?",
        "I don't play tennis every Sunday."
    ],
    ("Present", "Progressive"): [
        "I am playing tennis right now.",
        "Are you playing tennis right now?",
        "I'm not playing tennis right now."
    ],
    ("Present", "Perfect"): [
        "I have played tennis for five years.",
        "Have you played tennis for five years?",
        "I haven't played tennis for five years."
    ],
    ("Present", "Perfect Progressive"): [
        "I have been playing tennis since morning.",
        "Have you been playing tennis since morning?",
        "I haven't been playing tennis since morning."
    ],
    ("Past", "Simple"): [
        "I played tennis yesterday.",
        "Did you play tennis yesterday?",
        "I didn't play tennis yesterday."
    ],
    ("Past", "Progressive"): [
        "I was playing tennis when it started raining.",
        "Were you playing tennis when it started raining?",
        "I wasn't playing tennis when it started raining."
    ],
    ("Past", "Perfect"): [
        "I had played tennis before I got tired.",
        "Had you played tennis before you got tired?",
        "I hadn't played tennis before I got tired."
    ],
    ("Past", "Perfect Progressive"): [
        "I had been playing tennis for two hours when it started raining.",
        "Had you been playing tennis for two hours when it started raining?",
        "I hadn't been playing tennis for two hours when it started raining."
    ],
    ("Future", "Simple"): [
        "I will play tennis tomorrow.",
        "Will you play tennis tomorrow?",
        "I won't play tennis tomorrow."
    ],
    ("Future", "Progressive"): [
        "I will be playing tennis at 3 PM tomorrow.",
        "Will you be playing tennis at 3 PM tomorrow?",
        "I won't be playing tennis at 3 PM tomorrow."
    ],
    ("Future", "Perfect"): [
        "I will have played tennis for an hour by the time you arrive.",
        "Will you have played tennis for an hour by the time I arrive?",
        "I won't have played tennis for an hour by the time you arrive."
    ],
    ("Future", "Perfect Progressive"): [
        "I will have been playing tennis for three hours by the time the match ends.",
        "Will you have been playing tennis for three hours by the time the match ends?",
        "I won't have been playing tennis for three hours by the time the match ends."
    ],
    ("Future in the Past", "Simple"): [
        "I knew I would play tennis the next day.",
        "Did you know you would play tennis the next day?",
        "I didn't know I would play tennis the next day."
    ],
    ("Future in the Past", "Progressive"): [
        "I thought I would be playing tennis at this time.",
        "Did you think you would be playing tennis at this time?",
        "I didn't think I would be playing tennis at this time."
    ],
    ("Future in the Past", "Perfect"): [
        "I believed I would have played tennis before they arrived.",
        "Did you believe you would have played tennis before they arrived?",
        "I didn't believe I would have played tennis before they arrived."
    ],
    ("Future in the Past", "Perfect Progressive"): [
        "I hoped I would have been playing tennis for a while before the party started.",
        "Did you hope you would have been playing tennis for a while before the party started?",
        "I didn't hope I would have been playing tennis for a while before the party started."
    ]
}

# Function to print a cell of examples
def print_cell(examples):
    for example in examples:
        print(example)
    print()

# Print the table
for row_label, row_tense in zip(tenses[1], tenses[1]):
    print(f"\t\t{row_label}", end="\t" * 4)
print()

for col_label, col_tense in zip(tenses[0], tenses[0]):
    print(col_label, end="\t")
    for row_label, row_tense in zip(tenses[1], tenses[1]):
        print_cell(examples[(row_label, col_label)])


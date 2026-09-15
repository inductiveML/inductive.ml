export const monarchPrompts = [
  {label: 'Explain simply', text: 'Explain why the sky is blue in three sentences.'},
  {label: 'Extract JSON', text: 'Extract the information as JSON with keys "name", "age", and "city". Return only a fenced JSON code block. Text: Ada is 28 years old and lives in Bangalore.'},
  {label: 'Summarize', text: 'Summarize this in three bullet points: Our library opens at 9 AM on weekdays and 10 AM on weekends. Members can borrow five books at a time for two weeks. Late returns cost one dollar per day. Study rooms can be reserved online.'},
  {label: 'Rewrite clearly', text: 'Rewrite this message to sound friendly and professional: hey, the report is late again. send it by 3 pm today because we need it for the meeting.'},
  {label: 'Compare ideas', text: 'Compare a CPU and a GPU in a small Markdown table with columns Feature, CPU, and GPU. Include three rows.'},
  {label: 'Classify sentiment', text: 'Classify each review as Positive, Negative, or Neutral. Return a numbered list.\n1. The coffee was excellent and the staff were kind.\n2. My order arrived cold and an hour late.\n3. The shop is next to the train station.'},
  {label: 'Make a checklist', text: 'Make a short Markdown checklist of five things to pack for a day hike. Use unchecked boxes.'},
  {label: 'Write Python', text: 'Write a Python function that returns the largest number in a non-empty list. Use a fenced code block, then give one example.'},
  {label: 'Translate', text: 'Translate this sentence into French, Spanish, and Hindi. Use a bullet list: Thank you for your help. Have a wonderful day!'},
] as const;

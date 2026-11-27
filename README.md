✍️ Dysgraphia Handwriting Screening

📸 Upload a photo of handwriting ✅ get a quick screening result.

⚠️ This is a screening aid only. It is NOT a medical diagnosis.

🔍 OVERVIEW
This project is a handwriting screening tool. A person uploads a photo of handwriting, and the app says whether it shows a pattern linked to dysgraphia, a difficulty with writing. It also shows how likely that result is, and which of its two AI models made the decision.

🎯 WHAT THIS IS SOLVING
Dysgraphia is often noticed late. Checking handwriting by hand takes a trained specialist's time, and opinions can vary from one person to another. This tool gives a quick first check from just a photo, with no special equipment. It helps decide whether someone should be looked at more closely. It does not give a diagnosis.

⚙️ HOW THIS IS IMPLEMENTED
📷 Photo cleaning: the app removes the effect of lighting, tilt, paper lines and camera angle.
✂️ Splitting: the writing is cut into small pieces, one per line of text.
🤖 Two AI models study the pieces:
🖼️ CNN: looks at the writing like a picture.
📖 RNN: reads the writing line by line, like a person reading.
🏆 Choosing the winner: both models were tested on children they had never seen. The CNN did better, so the app uses it for the final answer.
🏷️ Transparency: the result page always shows which model decided, so nothing is hidden.
📊 Extra context: the app also lists simple handwriting qualities, such as spacing and letter size, so you can see what stood out. These do not decide the result.


🧭 WHAT YOU SEE WHEN YOU USE IT
You upload one photo.
You get a YES or NO.
You see how likely the result is, and a warning if it is a close call.
You see a label such as "Classified by CNN".
You see a short list of handwriting qualities marked typical or unusual.


💡 USE CASES
👩‍🏫 Teachers and parents: an early warning before seeking professional help.
🏫 Schools: quick screening of many students.
🩺 Therapists: a supporting tool alongside their own assessment.
🎓 Students and researchers: a working example of AI in healthcare.


🚀 FUTURE SCOPE
📚 Much more data from many more children, of different ages, languages and schools.
🏥 Testing on clinically confirmed cases.
🧑 Support for adult handwriting.
📱 A phone app.
📈 Tracking a child's progress over time.
🖊️ Digital pens to capture pressure and speed.
📄 A clear, easy-to-read report for teachers and parents.


📏 HOW WELL DOES IT WORK?
On children it had never seen, the CNN was right roughly 87 times out of 100 on average across both groups, and the RNN roughly 80 times out of 100. About 1 in 5 typical writers was wrongly flagged. The group of children it learned from was small, so these numbers are only a rough guide.


❓ QUICK QUESTIONS
Does a YES mean the person has dysgraphia? No. It means the writing looks like examples linked to it, and a professional should take a closer look.

Does a NO mean everything is fine? No. The tool can miss cases, so worries should still be raised with a professional.

Can I trust it with any handwriting? It works best on handwriting similar to what it learned from. Other languages, adult handwriting or drawings may give weaker results.

Is my photo kept? The app only looks at the photo while you are using it and does not save it to disk. Wherever you run it, still follow your own privacy rules, especially with children's photos.


⚠️ IMPORTANT NOTE
This tool is for screening only and is not a medical diagnosis. It learned from a small group of children, so it can make mistakes. Only a qualified professional can confirm dysgraphia.

Smart Patient Triage System

This project solves a real life problem of emergency clinics about how
to prioritize patients in a busy clinic. Instead of a simple first come 
first serve basis this system uses a Decision Tree Classifier to predict
the urgency of the patients injury(on the basis of body temp, age and pain level) 
and a Priority Queue(Heapq) to manage the waiting list.

This project works in three different stages:
  
  1) In the first stage we generate a dataset of 10,000 patients with random
     data on thier temperatures, pain level and ages.

  2) In the second stage we train the AI with machine learning. A scikit-learn
     Decision Tree studies the 10,00 records. It studies the patterns so it can
     predict the priority value of the new patients based on thier data.

  3) In the third stage the AI predicts the priority value of the new patients and
     add them to a heapq(Priority queue). The code then searches the top of the heapq
     to make sure that the most critical patient is taken next by the doctor.

Tech Used:

1) Language: Python 3.x
2) Data Handling: Pandas, NumPy
3) Machine Learning: Scikit-learn (Decision Tree Classifier)
4) Search Strategy: Heapq

Code Breakdown:

1) Data Representation:
   The training phase comprises of DataFrame(dF). It is a tabular data structure containing
   Body Temp(97.0-105.0), Pain Level(1-10), Age(1-95) and Priority(0:Routine, 1:Urgent, 2:Emergency)

2) ML part:
   We use a decision tree to predict the priority value of new patients.

3) Search Algorithm:
   We do a priority search using heap.
   In the case of two patients having the same priority value we store a timestamp
   so that the patient who arrives first is treated first in this case.

Getting Started:

Make sure you have all the necessary libraries installed.
Copy the code into a file and then run it. 
The example output is attached in this repository.

Future Plans:

1) GUI: Can build a gui for it to meake it more interacctive.
2) Real dataset: Use a real dataset from a hospital to train the ml model.
3) Input system: to add the patients in a more interactive method.

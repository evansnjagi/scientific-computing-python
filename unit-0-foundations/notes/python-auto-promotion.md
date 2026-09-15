# Python Core Data Types
> List, tuple and a dictionary (key-value pairs, for labelled data)

## Question
Explain what Python's automatic type promotion is, using the total/average example, and why explicit typing still matters as a general habit in scientific computing.

## Solution
In Python, when an integer is mixed with a float, you don't need to explicitly type cast the integer. Python automatically promotes the integer into a float during the operation. However, it's good practice to be explicit about types e.g. initializing `totals` as $0.0$ instead of $0$ makes the intent of the coder clearer, event though Python doen't need it here. This habit becomes essential in stricter numerical libraries like NumPy, where types are not always promoted automatically. 
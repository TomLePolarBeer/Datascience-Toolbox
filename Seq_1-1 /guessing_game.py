import numpy as np

number = np.random.randint(0, 100)
guess = int(input("Choisis un nombre entre 0 et 100 : "))

while number != guess :
    if guess > number :
        print("Plus petit")
    else :
        print("Plus grand")

    guess = int(input("Choisis un nombre entre 0 et 100 : "))

print("Tu as trouvé le bon nombre !")
    



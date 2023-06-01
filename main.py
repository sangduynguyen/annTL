# main.py
import sys
try:
 x = int(input('Choose task (baseline:0, GOK_ANN:1, GOKs1D:2, GOKs2D:3, GOKs3D:4): '))
 if x == 0:
    print('You have chosen baseline\nPlease go to the csv file and select the TL curve to analyze!')
    import baseline
 elif x == 1:
    print('You have chosen the model GOK_ANN\nPlease click Open file .csv and select the TL curve to analyze!')
    import GOK_ANN
 elif x == 2:
    print('You have chosen GOKs1D\nPlease click Open file .csv and select the TL curves to analyze!')
    import GOKs1D
 elif x == 3:
    print('You have chosen GOKs2D\nPlease click Open file .csv and select the TL curves to analyze!')
    import GOKs2D
 elif x == 3:
    print('You have chosen GOKs2D\nPlease click Open file .csv and select the TL curves to analyze!')
    import GOKs3D
 else:
    print('You have not selected any model, please go to main function to run it again')
except:
 print("Finish!")
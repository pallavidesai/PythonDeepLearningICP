#Write a python program to print first letter of name using star symbols.

OutputString="";
for Horizontal in range(0,7):
    for Vertical in range(0,7):
        if (Vertical == 1 or ((Horizontal == 0 or Horizontal == 3) and Vertical > 0 and Vertical < 5) or ((Vertical == 5 or Vertical == 1) and (Horizontal == 1 or Horizontal == 2))):
            #Write star
            OutputString=OutputString+"*"
        else:
            #Write Nothing
            OutputString=OutputString+" "
    #Go to new line
    OutputString=OutputString+"\n"
print(OutputString);
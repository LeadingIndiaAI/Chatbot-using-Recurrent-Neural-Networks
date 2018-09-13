from tkinter import *
from tkinter import scrolledtext
from test_seq2seq import ChatBot
from tkinter import LEFT,RIGHT,TOP,BOTTOM

#Calling Class for chat prediction
ob = ChatBot()

#main display chat window 
window = Tk()
window.title("ChatCraZie")
window.geometry('550x450')

#top frame to display the chat history
frame1 = Frame(window, class_="TOP")
frame1.pack(expand=True, fill=BOTH)

#text area with scroll bar
textarea = Text(frame1, state=DISABLED)
vsb = Scrollbar(frame1, takefocus=
                0, command=textarea.yview)
vsb.pack(side=RIGHT, fill=Y)
textarea.pack(side=RIGHT, expand=YES, fill=BOTH)
textarea["yscrollcommand"]=vsb.set

#bottom frame to display current user question text box 
frame2 = Frame(window, class_="Chatbox_Entry")
frame2.pack(fill=X, anchor=N)

lbl = Label(frame2, text="User : ")
lbl.pack(side=LEFT)
 

def bind_entry(self, event, handler):
    txt.bind(event, handler)

def clicked(event): 
    #to automate the scrollbar action downward according to the text
    relative_position_of_scrollbar = vsb.get()[1]
    res =txt.get() 
    #function call
    ans = ob.test_run(res)
    pr="Human : " + res + "\n" + "ChatBot : " + ans + "\n"
    #the state of the textarea is normalto write the text to the top area in the interface
    textarea.config(state=NORMAL)
    textarea.insert(END,pr)
    #it is again disabled to avoid the user modifications in the history
    textarea.config(state=DISABLED)
    txt.delete(0,END)
    if relative_position_of_scrollbar == 1:
        textarea.yview_moveto(1)
    txt.focus()

txt = Entry(frame2,width=70)
txt.pack(side=LEFT,expand=YES, fill=BOTH)
txt.focus()
txt.bind("<Return>", clicked)

window.mainloop()
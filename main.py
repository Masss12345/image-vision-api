from flask import *

from detect_text import *
#from toxic_comment_detection import *

app=Flask(__name__)
app.config['SECRET_KEY']='toxiccomment'
app.config['UPLOAD_FOLDER']="C:/Users/MaSsS/OneDrive/Desktop/cloud project/TextImage"

@app.route('/',methods=['GET','POST'])
def homepage():
    import os
    if request.method=='POST':
        file=request.files['file']
        file_loc=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(file_loc)
        text=detect_text(file_loc)
        return render_template('result.html',data=text)
    return render_template('homepage.html')

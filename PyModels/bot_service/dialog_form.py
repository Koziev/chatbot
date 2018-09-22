from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired

class DialogForm(FlaskForm):
    phrases = StringField('phrases', widget=TextArea())
    utterance = StringField('utterance', validators=[DataRequired()])
    submit = SubmitField('Send')

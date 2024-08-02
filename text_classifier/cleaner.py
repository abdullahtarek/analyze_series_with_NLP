from bs4 import BeautifulSoup

class Cleaner():
  def __init__(self):
    pass
  def put_line_breaks(self,text):
    text = text.replace('</p>','</p>\n')
    return text
  def remove_html_tags(self,text):
    cleantext = BeautifulSoup(text, "lxml").text
    return cleantext
  def clean(self,text):
    text = self.put_line_breaks(text)
    text = self.remove_html_tags(text)
    return text
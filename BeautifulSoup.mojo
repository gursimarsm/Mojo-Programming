html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

"""

from PythonInterface import Python

let bs4 = Python.import_module("bs4")
let builtins = Python.import_module("builtins")
let bprint = builtins.print


let soup = bs4.BeautifulSoup(html_doc)

let result = soup.prettify()

builtins.print(result)

bprint("print end!")

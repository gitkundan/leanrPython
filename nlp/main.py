import re
from nltk.tokenize import regexp_tokenize

my_string = r"SOLDIER #1: Found them? In Mercea? The coconut's tropical!"
pattern=r'(\w+|\?|!)'
# matches=re.findall(pattern,my_string)
matches=regexp_tokenize(my_string,pattern)
print(matches)
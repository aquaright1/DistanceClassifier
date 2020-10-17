from checks import *

X, y = get_data("ecoli.data", "cp", "species")

print(set(y))

pdf_compare("ecoli", X, y, False)
pdf_compare("ecoli", X, y, True)

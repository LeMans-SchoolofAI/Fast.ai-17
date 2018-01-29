import sys,json

inputfile = sys.argv[1] 
outputfile = (str(inputfile) + ".py")
print("nb path: {}".format(inputfile))
print("output file: {}".format(outputfile))

#raise

#f = open(sys.argv[1], 'r') #input.ipynb
f = open(inputfile, 'r') 
of = open(outputfile, 'w')
j = json.load(f)
if j["nbformat"] >=4:
        for i,cell in enumerate(j["cells"]):
                of.write("#cell "+str(i)+"\n")
                for line in cell["source"]:
                        of.write(line)
                of.write('\n\n')
else:
        for i,cell in enumerate(j["worksheets"][0]["cells"]):
                of.write("#cell "+str(i)+"\n")
                for line in cell["input"]:
                        of.write(line)
                of.write('\n\n')

of.close()


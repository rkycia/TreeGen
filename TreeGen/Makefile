
FILE = Main.py
LIB = Generator.py ProbabilityTree.py

run: ${FILE}
	python3 ${FILE}

uml: ${FILE}
	pyreverse -S -A  -o png ${LIB}

#----------------------------------------------------------------------
#Generate documentation
Generate-doc:	
	-doxygen -g 
	-echo 'PROJECT_NAME = "ProbabilityTreeMC"' >> Doxyfile
	-echo 'RECURSIVE = YES' >> Doxyfile
	-echo 'SOURCE_BROWSER = YES' >> Doxyfile
	-echo 'INLINE_SOURCES = YES' >> Doxyfile
	-echo 'REFERENCES_LINK_SOURCE = YES' >> Doxyfile
	-echo 'GENERATE_RTF = YES' >> Doxyfile
	-echo 'GENERATE_MAN = YES' >> Doxyfile
	-doxygen Doxyfile
	-firefox ./html/index.html

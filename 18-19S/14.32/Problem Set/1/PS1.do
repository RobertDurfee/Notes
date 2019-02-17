* Clear the current dataset
clear

* Log the output to a file
log using PS1.log, replace

* Change working directory to problem set 1 folder
cd "/afs/athena.mit.edu/user/r/b/rbdurfee/Documents/Notes/18-19S/14.32/Problem Set/1"

* Download and load the data file
copy "http://economics.mit.edu/files/397" asciiqob.zip,replace
unzipfile asciiqob.zip
infile lwklywge educ yob qob pob using asciiqob.txt

* Save loaded data in Stata format
save ak91.dta,replace

* Regress log weekly wage with education
regress lwklywge educ

* Summarize log weekly wage and education
sum lwklywge educ

* Stop logging
log close

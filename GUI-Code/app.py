from flask import Flask,render_template,url_for,request
from rake import *
from sentAn import *
from array import *
import random
app = Flask(__name__)


@app.route('/')
def home():
	return render_template('section1.html')


@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		text = request.form['comment']

		inputMatrix = getSentenceMatrix(text)
		predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
		if (predictedSentiment[0] > predictedSentiment[1]):
		    sa = "Positive"
		else:
		    sa = "Negative"
		# RAKE part start
        sentenceList = split_sentences(text)
        stoppath = "SmartStoplist.txt"  #SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
        stopwordpattern = build_stop_word_regex(stoppath)
        phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)
        wordscores = calculate_word_scores(phraseList)

        keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)
        if debug: print keywordcandidates

        sortedKeywords = sorted(keywordcandidates.iteritems(), key=operator.itemgetter(1), reverse=True)
        if debug: print sortedKeywords

        totalKeywords = len(sortedKeywords)
        rake = Rake("SmartStoplist.txt")
        keywords = rake.run(text)
		# RAKE part end

	return render_template('section2.html',prediction = keywords, sentiment = sa, text = text)


@app.route('/nst',methods=['POST'])
def nst():
	if request.method == 'POST':
		gen_key = request.form.getlist('keyword')
		sa2 = request.form['sentiment']
	return render_template('section3.html', key = gen_key, sentiment = sa2)


@app.route('/final')
def final():
	return render_template('section4.html')


# To start the page without dealing with command line
if __name__== '__main__':
    app.run(debug=True)
# Beware that the __name__ isnt always equal to __main__
# and this is only true if we are running this file directly

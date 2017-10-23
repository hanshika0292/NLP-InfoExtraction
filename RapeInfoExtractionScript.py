import newspaper
import nltk
import numpy as np
from newspaper import Article
from nltk.collocations import *
from nltk.tokenize import word_tokenize
urls = ['http://www.news.com.au/entertainment/tv/current-affairs/woman-who-was-gang-raped-as-a-teenager-calls-to-stop-victim-blaming/news-story/3cae20f8c68a4c7e1336adba64dffcff','http://www.telegraph.co.uk/news/2017/10/21/italian-actress-accused-harvey-weinstein-rape-leaves-country/','http://www.france24.com/en/20171021-france-tariq-ramadan-muslim-scholar-accused-rape-sexual-assault-henda-ayari-ex-salafist','http://variety.com/2017/music/news/marilyn-manson-bassist-rape-1202595930/','http://www.bbc.com/news/world-asia-india-41335582?intlink_from_url=http://www.bbc.com/news/topics/a4741524-eaa1-451a-b5d1-98028e965367/rape-in-india&link_location=live-reporting-story','https://www.nytimes.com/2017/10/21/opinion/high-school-student-letters.html','http://www.bbc.com/news/world-europe-41704759'];
articles = [None] * len(urls);
for url in urls:
	ind = urls.index(url);
	articles[ind]=Article(url,language='en');
	articles[ind].download();
	articles[ind].parse();
for article in articles:
        print("\n");
        print("Article NO - ",articles.index(article)+1);
        print("Topic - ",article.title);
        article.nlp();
        print("Main Keywords - ",article.keywords);
        rapeNews = False;
        if "rape" in article.keywords or "raped" in article.keywords:
                rapeNews = True;
                print("Its a RAPE Article");
                trigram_measures = nltk.collocations.TrigramAssocMeasures();
                finder = TrigramCollocationFinder.from_words(word_tokenize(article.title));
                rape_filter = lambda *w: 'rape' not in w;
                finder.apply_freq_filter(3);
                finder.apply_ngram_filter(rape_filter);
                print(finder.nbest(trigram_measures.likelihood_ratio, 1));

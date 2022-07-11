#sigmorphon shared task 2022

#Swedish/Norwegian Nynorsk		swe/nno
#German/Dutch						ger/dut
#Italian/Romanian					ita/rum
#Ukrainian/Belarusian			ukr/bel
#SURPRISE/SURPRISE				gle/wel
#Tagalog/Cebuano					tgl/ceb
#Bengali/Assamese					ben/asm
#SURPRISE/SURPRISE				bur/shn
#Persian/Pashto					per/pus
#Thai/Eastern Lawa				tha/lwl

PFX=/mhdata/2022G2PST-main/

#for LANG in ger/dut

for LANG in ben/asm ger/dut ita/rum per/pus swe/nno tgl/ceb tha/lwl ukr/bel gle/wel bur/shn
do
	#get transfer and target lang names
	echo $LANG
	TOLANG=$(echo "$LANG" | cut -d/ -f1)
	FROMLANG=$(echo "$LANG" | cut -d/ -f2)
	echo $TOLANG and $FROMLANG
	#train on pure target language
	#--lexicon ${PFX}data/target_languages/${TOLANG}_100_train.tsv \
	#--lexicon ${PFX}data/target_languages/${TOLANG}_train.tsv \
	#--lexicon /mhdata/sigdata/phbg/${FROMLANG}_${TOLANG}.tsv \
	#--lexicon /mhdata/sigdata/aug/${TOLANG}.tsv \
	#add augmented
	#cp ${PFX}data/target_languages/${TOLANG}_100_train.tsv loc.tsv
	cp /mhdata/sigdata/phbg/${FROMLANG}_${TOLANG}.tsv loc.tsv
	cat /mhdata/sigdata/aug/${TOLANG}.tsv >> loc.tsv
	phonetisaurus-train \
		--lexicon loc.tsv \
		--model_prefix ${TOLANG} \
		--ngram_order 3 \
		--seq2_del \
		--seq1_del
	#massage dev list
	cut -f1 ${PFX}data/target_languages/${TOLANG}_dev.tsv > TMP.wlist
	cut -f2 ${PFX}data/target_languages/${TOLANG}_dev.tsv > ${TOLANG}.gold
	#test on dev list
	phonetisaurus-apply \
		--model train/${TOLANG}.fst \
		--word_list TMP.wlist \
		> ${TOLANG}.res
	rm TMP.wlist
	#massage dev results
	cut -f2 ${TOLANG}.res | paste ${TOLANG}.gold - > ${TOLANG}.all
	echo $TOLANG
	#get score for dev data
	python3 ${PFX}evaluation/evaluate.py ${TOLANG}.all
	#get test data results
	phonetisaurus-apply \
		--model train/${TOLANG}.fst \
		--word_list ${PFX}data/target_languages/${TOLANG}_test.tsv \
		> ${TOLANG}.test
done


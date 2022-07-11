#using opoennmt for sigmorphon2022

PFX=/mhdata/2022G2PST-main/

touch all.scores

#for LANGPAIR in ger/dut

for LANGPAIR in ben/asm ger/dut ita/rum per/pus swe/nno tgl/ceb tha/lwl ukr/bel gle/wel bur/shn
do
   #get transfer and target lang names
   echo $LANGPAIR
   LANG=$(echo "$LANGPAIR" | cut -d/ -f1)
   FROMLANG=$(echo "$LANGPAIR" | cut -d/ -f2)
	echo $LANG
	TRAIN=${PFX}data/target_languages/${LANG}_100_train.tsv
	#TRAIN=${PFX}data/target_languages/${LANG}_train.tsv
	#TRAIN=/mhdata/sigdata/phonorthodata/${FROMLANG}_${LANG}.tsv
	#TRAIN=/mhdata/sigdata/aug/${LANG}.tsv
	cp $TRAIN train.txt
	#cat ${PFX}data/target_languages/${LANG}_100_train.tsv >> train.txt
	#cat /mhdata/sigdata/phbg/${FROMLANG}_${LANG}.tsv >> train.txt
	#cut -f2 $TRAIN > tgt-train.txt
	#cut -f1 $TRAIN > TMPsrc-train.txt
	cut -f2 train.txt > tgt-train.txt
	cut -f1 train.txt > TMPsrc-train.txt
	rm train.txt
	python fix.py TMPsrc-train.txt > src-train.txt
	rm TMPsrc-train.txt
	DEV=${PFX}data/target_languages/${LANG}_dev.tsv
	cut -f2 $DEV > tgt-val.txt
	cut -f1 $DEV > TMPsrc-val.txt
	python fix.py TMPsrc-val.txt > src-val.txt
	rm TMPsrc-val.txt
	onmt_build_vocab -config sig2.yaml -n_sample 10000
	#onmt_train -config small.yaml -share_vocab
	onmt_train -config sig2.yaml
	onmt_translate \
		-model run/model_step_1000.pt \
		-src src-val.txt \
		-output pred100.txt \
		-gpu 0
	paste tgt-val.txt pred100.txt > res.txt
	echo ${LANG} >> all.scores
	python ${PFX}evaluation/evaluate.py res.txt >> all.scores
	TEST=${PFX}data/target_languages/${LANG}_test.tsv
	python fix.py $TEST > src-test.txt
	onmt_translate \
		-model run/model_step_1000.pt \
		-src src-test.txt \
		-output TMP${LANG}.txt \
		-gpu 0
	paste src-test.txt TMP${LANG}.txt > ${LANG}.test
	rm -r run
	rm *.txt
done


#!/bin/bash

#for n_players in 500 1000 2000 5000
for n_players in 500 1000 2000 5000
do
    for sample_method in 'Bern' 'BTL'
    #for sample_method in 'Bern'
    do
        #for rank_method in 'RC_org' 'RC' 'Borda'
        #for rank_method in 'RC' 'Borda'
        for rank_method in 'MLE' 'RC' 'Borda'
        do
            python rc.py --n_players $n_players --sample_method $sample_method \
                --rank_method $rank_method
        done
    done
done


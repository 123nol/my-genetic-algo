(= (pop-size) 4)
(= (genes) 8)
(= (generations) 30)
(= (elite-count) 3)
(= (crossover-rate) 0.9)
(= (mutation-rate) 0.1)
(= (initial-mutation-std) 0.5)
(= (mutation-decay) 0.95)
(= (initial-sbx-eta) 2)
(= (sbx-eta-growth) 1.05)

(= (input-A) (0.9 0.9 0.0 0.8 0.2 0.9 0.7 0.7))
(= (input-B) (0.8 0.2 1.0 0.3 0.9 0.4 0.6 0.3))

(= (zip-lists $lst1 $lst2 $acc)
   (if (or (== $lst1 ()) (== $lst2 ()))
       $acc
       (let* (
           ($head1 (car-atom $lst1))
           ($head2 (car-atom $lst2))
           ($tail1 (cdr-atom $lst1))
           ($tail2 (cdr-atom $lst2))
           ($pair ($head1 $head2))
           
         )
         (zip-lists $tail1 $tail2 (union-atom $acc ($pair)) )
       )
   )
)

(= (slice-atom $lst $start $end $acc)
   (if (or (== $lst ()) (>= $start $end))
       $acc
       (let* (
           ($el (index-atom $lst $start))
         )
         (slice-atom $lst (+ $start 1) $end (union-atom $acc ($el)))
       )
   )
)

; Convenience wrapper: (slice-atom-simple $lst $start $end)

(= (slice-atom $lst $start $end)
   (slice-atom $lst $start $end ())
)


(= (count-list $lst $acc)
   (if (== $lst ()) 
       $acc
       (count-list (cdr-atom $lst) (+ $acc 1))))

(=(cap $val) (if (< $val 0) 0 (if (> $val 1) 1 $val)))

; Helper: emergence = c - max(a, b)
;(: emergence-list (-> (List $c) (List $a) (List $b) (List $r) (List $r)))



(= (random-individual-helper $genes $count $acc)
   (if (== $count $genes)
       $acc
       (let* (
           ($val ((py-atom random.uniform) 0 1))
         )
         (random-individual-helper $genes (+ $count 1) (union-atom $acc ($val)))
       )
   )
)

;(: random-individual (-> Nat (List Number)))
(= (random-individual $genes)
   (random-individual-helper $genes 0 ())
)

(= (initialize-population-helper $pop-size $genes $acc)
   (if (== $pop-size 0)
       $acc
       (let* (
           ($individual (random-individual $genes))
         )
         (initialize-population-helper (- $pop-size 1) $genes (union-atom $acc ($individual)))
       )
   )
)

(= (initialize-population)
   (initialize-population-helper (pop-size) (genes) ())
)
;!(initialize-population)


(= (emergence-list $c $a $b $acc)
   (if (or (or (== $c ()) (== $a ())) (== $b ()))
       $acc
       (let* (
           ($ec (car-atom $c))
           ($ea (car-atom $a))
           ($eb (car-atom $b))
           ($em1 (- $ec (max-atom ($ea $eb))))
           ($em (if (< $em1 0) 0 $em1))
         )
         (emergence-list (cdr-atom $c) (cdr-atom $a) (cdr-atom $b) (union-atom $acc ($em)))
       )
   )
)


(= (contrib-list $a $b $e $acc)
   (if (or (or (== $a ()) (== $b ())) (== $e ()))
       $acc
       (let* (
           ($ea (car-atom $a))
           ($eb (car-atom $b))
           ($ee (car-atom $e))
           ($contrib (* (min-atom ($ea $eb)) $ee))
         )
         (contrib-list (cdr-atom $a) (cdr-atom $b) (cdr-atom $e) (union-atom $acc ($contrib)))
       )
   )
)

(= (fitness $candidate )
   (let* (
       
       ($genes-count (genes))
       ($emergence-clamped (emergence-list $candidate (input-A) (input-B) ()))
      
       ($contributions (contrib-list (input-A) (input-B) $emergence-clamped ()))
       ($total (foldl-atom $contributions 0 $acc $x (+ $acc $x)))
       ($score (/ $total $genes-count))
     )
     (if (> $score 1.0) 1.0 $score)
   )
)
;!(fitness (0.9 0.9 0.9 0.9 0.9 0.5 0.5 0.5) )


(= (sbx-crossover $p1 $p2 )
   (let* (
       ($rand-cross ((py-atom random.random)))
     )
     (if (> $rand-cross (crossover-rate))
         ($p1 $p2)
         (sbx-crossover-helper $p1 $p2 (initial-sbx-eta) () ())
     )
   )
)


(= (sbx-crossover-helper $p1 $p2 $eta $child1 $child2)
   (if (or (== $p1 ()) (== $p2 ()))
       ($child1 $child2)
       (let* (
           ($x1 (car-atom $p1))
           ($x2 (car-atom $p2))
           ($rest1 (cdr-atom $p1))
           ($rest2 (cdr-atom $p2))
           ($rand ((py-atom random.random)))
         )
         (if (<= $rand 0.5)
             (if (> (abs-math (- $x1 $x2)) (pow-math 10 -14))
                 (let* (
                     ($minx (min-atom ($x1 $x2)))
                     ($maxx (max-atom ($x1 $x2)))
                     ($rand2 ((py-atom random.random)))
                     ($beta (+ 1.0 (/ (* 2.0 $minx)  (- $maxx $minx))))
                     ($alpha (- 2.0 (pow-math $beta (* -1 (+ $eta 1.0)))))
                     ($cond (<= $rand2 (/ 1.0 $alpha)))
                     ($betaq (if $cond
                                 (pow-math (* $rand2 $alpha) (/ 1.0 (+ $eta 1.0)))
                                 (pow-math (/ 1.0 (- 2.0 (* $rand2 $alpha))) (/ 1.0 (+ $eta 1.0)))))
                     ($c1 (* 0.5 (- (+ $minx $maxx) (* $betaq (- $maxx $minx)))))
                     ($c2 (* 0.5 (+ (+ $minx $maxx) (* $betaq (- $maxx $minx)))))
                     ($c1-clamped (cap $c1))
                     ($c2-clamped (cap $c2))
                   )
                   (sbx-crossover-helper $rest1 $rest2 $eta (union-atom $child1 ($c1-clamped)) (union-atom $child2 ($c2-clamped)))
                 )
                 (sbx-crossover-helper $rest1 $rest2 $eta (union-atom $child1 ($x1)) (union-atom $child2 ($x2)))
             )
             (sbx-crossover-helper $rest1 $rest2 $eta (union-atom $child1 ($x1)) (union-atom $child2 ($x2)))
         )
       )
   )
)
;!(sbx-crossover (0.2 0.8 0.5) (0.7 0.3 0.9) )




;standard deviation is mutation-std its an input for random.gauss which chooses from a gaussian distribution with mean 0 and standard distribuiton 
(= (mutate $individual )
   (mutate-helper $individual (initial-mutation-std) (mutation-rate) ())
)


;(: mutate-helper (-> (List $a) $mutation-std $mutation-rate Nat (List $a) (List $a)))
(= (mutate-helper $individual $mutation-std $mutation-rate $acc)
   (if (== $individual ())
       $acc
       (let* (
           ($gene (car-atom $individual))
           ($rest (cdr-atom $individual))
           ($rand ((py-atom random.random)))
           ($mutated-gene (if (< $rand $mutation-rate)
                               (let* (
                                   ($noise ((py-atom random.gauss) 0 $mutation-std))
                                   ($new-gene (+ $gene $noise))
                                   ($clipped-gene (cap $new-gene))
                                 )
                                 $clipped-gene
                               )
                               $gene
                            )
           )
         )
         (mutate-helper $rest $mutation-std $mutation-rate  (union-atom $acc ($mutated-gene)))
       )
   )
)
; !(mutate (0.2 0.5 0.8) )

(=(check-cand $cand $rand $max)
  (let* ((($ind ($fit)) (decons-atom $cand))) (if (>$rand (/ $fit $max)) (empty) $ind))

)


(=(roulette-stoch-undeter $population $fitness)
(let* (
  ($rand ((py-atom random.random)))

($ziped (zip-lists $population $fitness ()))

($valid-parents (collapse (check-cand (superpose $ziped) $rand (max-atom $fitness  ) )) )

($len (count-list $valid-parents 0))

; ($point ((py-atom random.randint) 0 (- $len 1)))
($point (random-int &rng 0 (- $len 1)))

($random-valid-parent  (index-atom $population $point))
)
$random-valid-parent

  
  
  )

)

;!(roulette-stoch-undeter ((a) (b) (c) (d) (e) (f) (g)) (0.1 0.3 0.6 0.9 0.9 0.8 0.8))

(= (roulette-wheel $population $fitnesses)
   (let* (
       ($total-f (foldl-atom $fitnesses 0 $acc $x (+ $acc $x)))
       ($r ((py-atom random.uniform) 0 $total-f))
     )
     (roulette-wheel-helper $population $fitnesses 0.0 $r)
   )
)
(= (roulette-wheel-helper $population $fitnesses $cum $r)
   (if (or (== $population ()) (== $fitnesses ()))
       (empty) ; fallback if not found
       (let* (
           ($individual (car-atom $population))
           ($fit (car-atom $fitnesses))
           ($new-cum (+ $cum $fit))
         )
         (if (>= $new-cum $r)
             $individual
             (roulette-wheel-helper (cdr-atom $population) (cdr-atom $fitnesses) $new-cum $r)
         )
       )
   )
)
!(roulette-wheel ((a) (b) (c) (d)) (0.1 0.3 0.6 0.9))




(= (sp-crossover $p1 $p2 $cor)
   (let* (
       ($rand ((py-atom random.random)))
     )
     (if (> $rand (crossover-rate))
         ($p1 $p2)
         (let* (
             ($len (count-list $p1 0))
             ($point ((py-atom random.randint) 1 (- $len 1)))
             ($fh1 (slice-atom $p1 0 $point))
             ($sh1 (slice-atom $p2 $point $len))
             ($fh2 (slice-atom $p2 0 $point))
             ($sh2 (slice-atom $p1 $point $len))
             ($child1 (union-atom $fh1 $sh1))
             ($child2 (union-atom $fh2 $sh2))
           )
           ($child1 $child2)
         )
     )
   )
)
; !(sp-crossover (1 2 3 4) (5 6 7 8) 0.9)


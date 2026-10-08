# Head-view check: plain-English summary

**This is a student research prototype. It has not been validated on cryptic pregnancies and is not for medical use.**

## The problem
Our app estimates how many weeks pregnant someone is from an ultrasound picture of the baby's head, and combines that with one
question ("did you know you were pregnant?") to suggest whether a pregnancy was found late (about 20 weeks or later).
But the app gave a week number for *any* picture, even a baby's tummy, leg or chest. Its old "is this a normal head scan?" check
let about 1 in 4 of those through, and it also wrongly turned away about 1 in 4 real head scans.

## What we did
We built a second check that asks one simple question: "is this a picture of a baby's head?" We trained it on about 10,000
routine scans (head, abdomen, leg, chest, cervix and other views) plus a few hundred head scans from the dataset our age
model was built on. We kept a separate set of images that it never saw during training, and tested only on those. An image
now gets an age only if **both** the old check and the new check accept it.

## What we found
- **On the new test pictures (2,478 scans):** the new check accepts about 95% of real heads and wrongly accepts about 0.2% of
  non-head pictures. The old check accepted about 75% of heads and about 25% of non-heads.
- **On the original head-scan dataset (HC18):** it still accepts about 97% to 99% of heads, so the app keeps working on the data
  it was built on. (Our first version, trained without those head scans, wrongly refused about 1 in 4 of them.)
- **In the app:** non-head pictures that used to get an age and a "Cryptic" label (about 6 in 100) now get "No estimate" in almost
  every case (about 1 in 1,000 still slip through).
- A Poor-quality scan no longer shows an age at all; it says "No estimate" and explains why.
- You can switch the check to a more lenient setting (98%) that accepts more real heads but lets a few more non-head pictures through.

## Things to be careful about
- It was not tested on non-head pictures that look like our original dataset, because that dataset only has head scans.
- We could not exactly reproduce how the original dataset's scans were split; we say so openly in the results.
- It has only seen scans from a few machines in two hospitals plus the original dataset; other machines are untested.
- About 1 in 20 real heads are still turned away at the default setting. That is a trade-off we chose on purpose.
- No dataset we used contains confirmed cryptic pregnancies, so we cannot say how well the cryptic-pregnancy suggestion works.

## What is in the share folder
`02_testing_set.zip` (the pictures we tested on, with a table of what each check decided for each picture),
`RESULTS_SUMMARY.md` (the numbers with confidence ranges), `CITATIONS_AND_LICENSES.md` and `NOTICE.md` (credit to the dataset
creators, required by their CC BY 4.0 licence).

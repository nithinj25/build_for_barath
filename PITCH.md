# 3-minute spoken script

Open the live page and click through it once, a minute before you start, so the
first Lambda request is already warm. Speak the **bold** lines; the bracketed
lines are what you click. Roughly 450 words — about three minutes with pauses.

---

## 0:00 — The gap (25 s)

> **Every property crime in India gets an FIR. A serial burglar's ten cases sit
> in ten different registers — different stations, often different districts or
> states.**
>
> **CCTNS and ICJS already move that data between states. Cri-MAC already lets
> an officer search it. But all of that assumes somebody already suspects a
> link — a name, a vehicle number, a tip.**
>
> **Nobody computes: this burglary in Mysuru looks like that one, three weeks
> later, in the next district. The pipes exist. The inference doesn't. That is
> what we built.**

## 0:25 — Show it reading an FIR (45 s)

[ New FIR → Fill with a test FIR ]

> **This is an FIR as it was written. I'll let the system read it.**

[ point at the highlighted phrases ]

> **It has pulled out the crime type, the date, the place, and how it was done —
> no AI model, about ten milliseconds — and it marks every value it read, so the
> officer checks it rather than trusting it.**

[ Find matching FIRs ]

> **That just compared this one FIR against every FIR of its type — thousands of
> them, across four states — in under a second.**

[ read the top match off the screen ]

> **Top match: a few weeks apart, same tools, same way in. And it tells you how
> unusual that is — rarer than one in two lakh unrelated cases look this alike.**

## 1:10 — Why it says so (35 s)

[ Why this match → scroll the reasons ]

> **Every match is explained in words an officer can take to a supervisor: the
> time gap, each shared habit and how common it is, what differs, what was never
> recorded.**
>
> **You will not find a probability anywhere in this product. A true link works
> out to about zero point three percent, which reads as broken — and a confident
> percentage invites false certainty. So we say how rare the match is, measured
> on two lakh random unrelated pairs.**
>
> **The officer marks it Linked, Needs investigation, or Not linked. Every
> lookup and every decision is logged.**

## 1:45 — The part nobody asked for (35 s)

[ Leads ]

> **The system compares all forty-four thousand FIRs against each other, not
> because anyone asked. Today it has three hundred and sixty same-district
> leads. In testing, about one in three of these was a real serial offender —
> roughly seven thousand times better than reading case pairs at random.**

[ Series → open one ]

> **And it chains them: eight FIRs, three districts, one modus operandi, on a
> board with the string — the way an investigator would lay it out.**

## 2:20 — The honest part (25 s)

> **The data is synthetic, because no real FIR corpus exists to train on, and
> publishing one would be wrong. On offenders the model never saw, the true
> partner is in the top ten for twenty-eight percent of cases — out of fifteen
> thousand FIRs of that type.**
>
> **Links across state borders are our weakest spot, and the screen says so
> rather than hiding it. Everything we tried that failed is written down in the
> repository too.**

## 2:45 — Close (15 s)

> **It runs on one Lambda on the AWS free tier — about zero rupees a month — and
> the whole interface is in Hindi as well, because the officer using it sits in
> a district station, not a data centre.**
>
> **It never names a person. It narrows fifteen thousand FIRs to ten worth
> reading, and a human decides.**

---

## If you only get 30 seconds

> **Serial property crime is invisible across district and state lines, because
> nobody compares FIRs to each other. We do: every FIR against every other, by
> modus operandi and timing. It hands an officer ten cases worth reading out of
> fifteen thousand, with the reasons written out — and in testing, one in three
> of its strongest leads was a real serial offender. It runs on the AWS free
> tier and it never names a person.**

## Questions you will get

**"How accurate is it?"**
> Accuracy is the wrong measure — a true link is one pair in twenty-six
> thousand, so answering "never linked" is 99.996% accurate and useless. What
> matters is rank: the true partner is in the top ten for 28% of cases, 50% if
> the officer turns on "nearby first".

**"Is this real data?"**
> No. Synthetic FIRs generated to resemble four states' records, with ground
> truth so we can measure. Two of our gains — timing and distance — partly come
> from assumptions in that generator, and we say so in the findings.

**"Isn't this profiling?"**
> There is no caste, religion or community field anywhere in the system, and
> personal details never enter matching. It compares how crimes were committed,
> and the output is always a pair of case numbers plus reasons.

**"What does it cost to run?"**
> One Lambda, two DynamoDB tables, no API Gateway. Free tier covers it.

**"Why not just use AI / an LLM?"**
> For the text reading we tried one locally; a phrase dictionary was faster,
> costs nothing, and never invents a value that isn't in the FIR. For the
> matching itself, the model is a Fellegi–Sunter weighting with a logistic
> correction — which is why every match can be explained line by line.

## If the demo fails

Run `python scripts/serve_local.py --demo` — same page, same data, no internet.
Screenshots in `docs/` are the last resort. Say what you are showing and keep
moving; a calm fallback reads better than a scramble.

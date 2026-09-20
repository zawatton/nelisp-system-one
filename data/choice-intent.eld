;; -*- lisp-data -*-
;;
;; P2's Choice task: "which of these topics is this request about?"
;;
;; Twelve topics, twelve short requests each, 144 examples.  Four topics are
;; marked :held and are absent from training entirely -- neither their
;; sentences nor their option text is available while the head is fitted.  The
;; head is then asked to choose among those four, and among all twelve.
;;
;; That split is the point of the phase.  A head that classifies a state into
;; N fixed slots cannot be asked about a slot it was never trained on; the
;; option-scoring form of section 3.1 can, because every logit is a function
;; of the option's own embedding.  If held-out-topic accuracy collapses to
;; chance, the architecture argument fails and the doc has to say so.
;;
;; The option text never appears in its sentences.  "The application crashes
;; on startup" contains neither "software" nor "bug"; "I have had a fever
;; since Monday" contains neither "medical" nor "symptom".  Lexical overlap
;; would let a probe match strings and would make the transfer result mean
;; nothing.
;;
;; Requests are the kind of thing an intent router sees, which is also the
;; shape of the eventual anvil use in section 6's P4 -- though P4 remains out
;; of scope and nothing here is evidence about it.

(:question "Which topic is this request about?"
 :template "%s Topic:"
 :topics
 [(:name "weather forecast" :held nil)
  (:name "cooking recipe"   :held nil)
  (:name "car maintenance"  :held nil)
  (:name "medical symptom"  :held t)
  (:name "travel booking"   :held nil)
  (:name "software bug"     :held nil)
  (:name "legal question"   :held t)
  (:name "personal finance" :held nil)
  (:name "gardening advice" :held t)
  (:name "pet care"         :held nil)
  (:name "home repair"      :held nil)
  (:name "fitness training" :held t)]
 :examples
 [
  (:topic "weather forecast" :text "Will it rain here tomorrow afternoon?")
  (:topic "weather forecast" :text "How cold does it get in January?")
  (:topic "weather forecast" :text "Is the storm expected to reach the coast?")
  (:topic "weather forecast" :text "What is it going to be like at the weekend?")
  (:topic "weather forecast" :text "Should I expect snow on the pass?")
  (:topic "weather forecast" :text "How windy will it be tonight?")
  (:topic "weather forecast" :text "Is there a frost warning for tonight?")
  (:topic "weather forecast" :text "When will the heatwave break?")
  (:topic "weather forecast" :text "How much rain fell last month?")
  (:topic "weather forecast" :text "Will the fog clear by noon?")
  (:topic "weather forecast" :text "Is a hurricane heading this way?")
  (:topic "weather forecast" :text "What will the humidity be like?")

  (:topic "cooking recipe" :text "How long should I roast a chicken?")
  (:topic "cooking recipe" :text "What can I substitute for buttermilk?")
  (:topic "cooking recipe" :text "My bread did not rise properly.")
  (:topic "cooking recipe" :text "At what temperature do I bake this cake?")
  (:topic "cooking recipe" :text "How do I thicken a sauce?")
  (:topic "cooking recipe" :text "Should I salt the pasta water?")
  (:topic "cooking recipe" :text "How long do I marinate the beef?")
  (:topic "cooking recipe" :text "What herbs go with roasted lamb?")
  (:topic "cooking recipe" :text "My custard curdled while heating.")
  (:topic "cooking recipe" :text "How do I keep pastry from shrinking?")
  (:topic "cooking recipe" :text "Can I freeze this soup?")
  (:topic "cooking recipe" :text "How do I know the steak is done?")

  (:topic "car maintenance" :text "My brakes squeal when I slow down.")
  (:topic "car maintenance" :text "How often should I change the oil?")
  (:topic "car maintenance" :text "The engine light came on this morning.")
  (:topic "car maintenance" :text "What tyre pressure should I run?")
  (:topic "car maintenance" :text "The battery keeps going flat overnight.")
  (:topic "car maintenance" :text "There is a knocking sound when accelerating.")
  (:topic "car maintenance" :text "When should the timing belt be replaced?")
  (:topic "car maintenance" :text "The steering pulls to the left.")
  (:topic "car maintenance" :text "Coolant is leaking onto the driveway.")
  (:topic "car maintenance" :text "The clutch feels soft lately.")
  (:topic "car maintenance" :text "How do I check the brake fluid?")
  (:topic "car maintenance" :text "The exhaust is blowing smoke.")

  (:topic "medical symptom" :text "I have had a headache for three days.")
  (:topic "medical symptom" :text "My ankle swelled up after a fall.")
  (:topic "medical symptom" :text "There is a rash on my forearm.")
  (:topic "medical symptom" :text "I keep feeling dizzy when standing.")
  (:topic "medical symptom" :text "My throat hurts when swallowing.")
  (:topic "medical symptom" :text "I have been short of breath lately.")
  (:topic "medical symptom" :text "There is a lump under my jaw.")
  (:topic "medical symptom" :text "My vision blurs in the evening.")
  (:topic "medical symptom" :text "I have had a fever since Monday.")
  (:topic "medical symptom" :text "My back aches after sitting.")
  (:topic "medical symptom" :text "There is numbness in my fingers.")
  (:topic "medical symptom" :text "I cannot sleep because of coughing.")

  (:topic "travel booking" :text "I need a flight to Lisbon in May.")
  (:topic "travel booking" :text "Can I change my hotel reservation?")
  (:topic "travel booking" :text "What is the baggage allowance?")
  (:topic "travel booking" :text "Is there a direct train to Milan?")
  (:topic "travel booking" :text "I want to cancel my reservation.")
  (:topic "travel booking" :text "Do I need a visa for this trip?")
  (:topic "travel booking" :text "Can I get a seat with more legroom?")
  (:topic "travel booking" :text "What time does check-in close?")
  (:topic "travel booking" :text "Is the rental car insured?")
  (:topic "travel booking" :text "I missed my connection this morning.")
  (:topic "travel booking" :text "Are there refunds for delayed trains?")
  (:topic "travel booking" :text "Can I book a room for three nights?")

  (:topic "software bug" :text "The application crashes on startup.")
  (:topic "software bug" :text "The page returns a blank screen.")
  (:topic "software bug" :text "Saving a file throws an error.")
  (:topic "software bug" :text "The build fails after the last commit.")
  (:topic "software bug" :text "Login stopped working this morning.")
  (:topic "software bug" :text "The list shows duplicate entries.")
  (:topic "software bug" :text "Memory usage grows until it stalls.")
  (:topic "software bug" :text "The test passes locally but fails in CI.")
  (:topic "software bug" :text "Dates are displayed in the wrong order.")
  (:topic "software bug" :text "The export produces an empty file.")
  (:topic "software bug" :text "Scrolling freezes on long pages.")
  (:topic "software bug" :text "The update rolled back by itself.")

  (:topic "legal question" :text "Can my landlord raise the rent mid-lease?")
  (:topic "legal question" :text "What notice period does my contract require?")
  (:topic "legal question" :text "Is this clause enforceable?")
  (:topic "legal question" :text "Who is liable for the damage?")
  (:topic "legal question" :text "How long do I have to file a claim?")
  (:topic "legal question" :text "Can I break the agreement early?")
  (:topic "legal question" :text "Does this count as fair use?")
  (:topic "legal question" :text "What happens if I miss the deadline?")
  (:topic "legal question" :text "Am I entitled to severance pay?")
  (:topic "legal question" :text "Can they keep my deposit?")
  (:topic "legal question" :text "Is a verbal agreement binding here?")
  (:topic "legal question" :text "Do I need to register the trademark?")

  (:topic "personal finance" :text "Should I pay off the loan early?")
  (:topic "personal finance" :text "How much should I keep in savings?")
  (:topic "personal finance" :text "What fees does this account charge?")
  (:topic "personal finance" :text "My card was declined at the till.")
  (:topic "personal finance" :text "How do I set up a direct debit?")
  (:topic "personal finance" :text "Is this interest rate competitive?")
  (:topic "personal finance" :text "I want to move money to a pension.")
  (:topic "personal finance" :text "What is my credit limit?")
  (:topic "personal finance" :text "How do I dispute a charge?")
  (:topic "personal finance" :text "Should I fix my mortgage rate?")
  (:topic "personal finance" :text "My salary payment has not arrived.")
  (:topic "personal finance" :text "How do I budget for the year?")

  (:topic "gardening advice" :text "My tomato leaves are turning yellow.")
  (:topic "gardening advice" :text "When should I prune the apple tree?")
  (:topic "gardening advice" :text "The lawn has bare patches.")
  (:topic "gardening advice" :text "How often do I water seedlings?")
  (:topic "gardening advice" :text "Slugs are eating my lettuce.")
  (:topic "gardening advice" :text "Which compost suits acid-loving plants?")
  (:topic "gardening advice" :text "My roses have black spots.")
  (:topic "gardening advice" :text "When do I plant garlic?")
  (:topic "gardening advice" :text "The hedge has stopped growing.")
  (:topic "gardening advice" :text "Can I move this shrub in autumn?")
  (:topic "gardening advice" :text "The soil drains too slowly.")
  (:topic "gardening advice" :text "How do I overwinter these bulbs?")

  (:topic "pet care" :text "My dog has been scratching constantly.")
  (:topic "pet care" :text "How much should I feed a kitten?")
  (:topic "pet care" :text "The rabbit is not eating today.")
  (:topic "pet care" :text "When are the next vaccinations due?")
  (:topic "pet care" :text "My cat keeps being sick after meals.")
  (:topic "pet care" :text "How do I trim a dog's nails?")
  (:topic "pet care" :text "The puppy will not settle at night.")
  (:topic "pet care" :text "Is this plant poisonous to cats?")
  (:topic "pet care" :text "My parrot is plucking feathers.")
  (:topic "pet care" :text "How often should I worm the dog?")
  (:topic "pet care" :text "The hamster seems lethargic.")
  (:topic "pet care" :text "Can I take the cat on the train?")

  (:topic "home repair" :text "The tap drips even when closed.")
  (:topic "home repair" :text "A floorboard creaks near the door.")
  (:topic "home repair" :text "There is damp on the bedroom wall.")
  (:topic "home repair" :text "The radiator stays cold at the top.")
  (:topic "home repair" :text "A window will not latch properly.")
  (:topic "home repair" :text "The doorbell has stopped ringing.")
  (:topic "home repair" :text "There is a crack in the ceiling plaster.")
  (:topic "home repair" :text "The shower pressure has dropped.")
  (:topic "home repair" :text "A socket sparks when I plug in.")
  (:topic "home repair" :text "The gutter overflows in heavy rain.")
  (:topic "home repair" :text "The boiler keeps losing pressure.")
  (:topic "home repair" :text "A tile has come loose in the hall.")

  (:topic "fitness training" :text "How many rest days should I take?")
  (:topic "fitness training" :text "My knee hurts after running.")
  (:topic "fitness training" :text "What weight should I start squatting?")
  (:topic "fitness training" :text "How do I improve my sprint time?")
  (:topic "fitness training" :text "Should I stretch before or after?")
  (:topic "fitness training" :text "I have stopped making progress.")
  (:topic "fitness training" :text "How long until I see results?")
  (:topic "fitness training" :text "What is a good warm-up routine?")
  (:topic "fitness training" :text "My shoulder aches during overhead presses.")
  (:topic "fitness training" :text "How many sets should I do?")
  (:topic "fitness training" :text "I want to run a half marathon.")
  (:topic "fitness training" :text "Is this heart rate too high?")
  ])

class InstructionsHandler:
    def __init__(self):
        self.ate = {}
        self.aspe = {}
        self.aope = {}

    def load_instruction_set1(
        self,
    ):

        ################################# ATE #################################

        self.ate[
            "bos_instruct1"
        ] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        output: features, iChat, Photobooth, garage band
        Now complete the following example-
        input: """

        self.ate[
            "bos_instruct2"
        ] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored.
        output: menu
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting.
        output: food, menu, service, setting
        Now complete the following example-
        input: """
        self.ate["delim_instruct"] = ""
        self.ate["eos_instruct"] = " \noutput:"

        ################################# ASPE #################################

        self.aspe[
            "bos_instruct1"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life:positive, 
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
        Now complete the following example-
        input: """

        self.aspe[
            "bos_instruct2"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored.
        output: menu:positive
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting.
        output: food:positive, menu:positive, service:positive, setting:positive
        Now complete the following example-
        input: """
        self.aspe["delim_instruct"] = ""
        self.aspe["eos_instruct"] = " \noutput:"

        ################################# AOPE #################################

        self.aope[
            "bos_instruct1"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the corresponding opinion/describing terms. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life:good 
        Positive example 2-
        input: it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous.
        output: quality:high, GUI:killer, applications:good, use:easy 
        Now complete the following example-
        input: """

        self.aope[
            "bos_instruct2"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: Faan 's got a great concept but a little rough on the delivery .
        output: delivery:rough
        Positive example 2- 
        input: I just wonder how you can have such a delicious meal for such little money .
        output: meal:delicious, money:little
        Now complete the following example-
        input: """
        self.aope["delim_instruct"] = ""
        self.aope["eos_instruct"] = " \noutput:"

    def load_instruction_set2(
        self,
    ):

        ################################# ATE #################################

        self.ate[
            "bos_instruct1"
        ] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        output: features, iChat, Photobooth, garage band
        Negative example 1-
        input: Speaking of the browser, it too has problems.
        output: browser
        Negative example 2-
        input: The keyboard is too slick.
        output: keyboard
        Neutral example 1-
        input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
        output: battery
        Neutral example 2-
        input: Nightly my computer defrags itself and runs a virus scan.
        output: virus scan
        Now complete the following example-
        input: """

        self.ate[
            "bos_instruct2"
        ] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored.
        output: menu
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting.
        output: food, menu, service, setting
        Negative example 1-
        input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
        output: toast, mayonnaise, bacon, ingredients, plate
        Negative example 2-
        input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
        output: seats
        Neutral example 1-
        input: I asked for seltzer with lime, no ice.
        output: seltzer with lime
        Neutral example 2-
        input: They wouldnt even let me finish my glass of wine before offering another.
        output: glass of wine
        Now complete the following example-
        input: """
        self.ate["delim_instruct"] = ""
        self.ate["eos_instruct"] = " \noutput:"

        ################################# ASPE #################################

        self.aspe[
            "bos_instruct1"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life:positive, 
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
        Negative example 1-
        input: Speaking of the browser, it too has problems.
        output: browser:negative
        Negative example 2-
        input: The keyboard is too slick.
        output: keyboard:negative
        Neutral example 1-
        input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
        output: battery:neutral
        Neutral example 2-
        input: Nightly my computer defrags itself and runs a virus scan.
        output: virus scan:neutral
        Now complete the following example-
        input: """

        self.aspe[
            "bos_instruct2"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored.
        output: menu:positive
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting.
        output: food:positive, menu:positive, service:positive, setting:positive
        Negative example 1-
        input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
        output: toast:negative, mayonnaise:negative, bacon:negative, ingredients:negative, plate:negative
        Negative example 2-
        input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
        output: seats:negative
        Neutral example 1-
        input: I asked for seltzer with lime, no ice.
        output: seltzer with lime:neutral
        Neutral example 2-
        input: They wouldnt even let me finish my glass of wine before offering another.
        output: glass of wine:neutral
        Now complete the following example-
        input: """
        self.aspe["delim_instruct"] = ""
        self.aspe["eos_instruct"] = " \noutput:"

        ################################# AOPE #################################

        self.aope[
            "bos_instruct1"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the corresponding opinion/describing terms. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life:good 
        Positive example 2-
        input: it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous.
        output: quality:high, GUI:killer, applications:good, use:easy
        Negative example 1-
        input: A month or so ago , the freaking motherboard just died .
        output: motherboard:freaking, motherboard:freaking
        Negative example 2-
        input: I had always used PCs and been constantly frustrated by the crashing and the poorly designed operating systems that were never very intuitive .
        output: operating systems:poorly designed, operating systems:intuitive
        Neutral example 1-
        input: It has a 10 hour battery life when you 're doing web browsing and word editing , making it perfect for the classroom or office , and in terms of gaming and movie playing it 'll have a battery life of just over 5 hours .
        output: web browsing:perfect, word editing:perfect
        Neutral example 2-
        input: no complaints with their desktop , and maybe because it just sits on your desktop , and you do n't carry it around , which could jar the hard drive , or the motherboard .
        output: hard drive:jar, motherboard:jar
        Now complete the following example-
        input: """

        self.aope[
            "bos_instruct2"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: Faan 's got a great concept but a little rough on the delivery .
        output: delivery:rough
        Positive example 2- 
        input: I just wonder how you can have such a delicious meal for such little money .
        output: meal:delicious, money:little
        Negative example 1-
        input: From the terrible service , to the bland food , not to mention the unaccommodating managers , the overall experience was horrible .
        output: service:terrible, food:bland, managers:unaccommodating
        Negative example 2- 
        input: I had the Pad Thai and the noodles were sticky .
        output: Pad Thai:sticky, noodles:sticky
        Neutral example 1-
        input: The Dim Sum was so-so , but not spectacular .
        output: Dim Sum:so-so, Dim Sum:not spectacular
        Neutral example 2- 
        input: The location and ambience is Ok but the food is what makes up for it .
        output: mlocationeal:Ok, ambience:Ok
        Now complete the following example-
        input: """
        self.aope["delim_instruct"] = ""
        self.aope["eos_instruct"] = " \noutput:"

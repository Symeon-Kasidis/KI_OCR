from Levenshtein import distance as ds
import math
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import roc_curve, auc
import csv
import warnings

warnings.filterwarnings("ignore")

#helper
def read_and_normalize_file(filename: str, splitat = ";"):
    try:
        with open(filename, "r",  encoding="utf-8-sig") as file:
            content = file.read()
            # Remove newlines, tabs, and extra spaces
            normalized_content = re.sub(r"\s+", " ", content).strip()
            return [part.strip() for part in normalized_content.split(splitat) if part.strip()]
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# define ground truth and responses
ground_truth = "Mülltrennung 1st wichtig für die Umwel8 Ök010g1scher Fußabdruck hängt von v1eIen Fakt0ren ab. Ü8ergrößen pass3n n1cht 1mm3r in d4s St4ndardregal Gänseßchen setzen 1st m4nchm4l k0mpl1z1ert"

ground_truth_jargon = "Gänsefüßchen 0l1OQqPp ßÜöÄäüÖ 8BbG6 Müllsäcke 1l1L0Oo9g6B8 WwVvUu"

ground_truth_form = read_and_normalize_file("ground_truth_form.txt")[0]
#35
claude_responses = [
"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hägt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänseßchen setzten ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hägt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänseßchen setzten ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwel♥ Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänseßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänsefüßchen setzten ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal  Gänseßchen setzten ist manchmal kompliziert",
]
#35
claude_jargon_response = [
"Gänsefüßchen 01102qPp ßÜöÄäüÖ 8Bb66 Müllsäcke 11L000g6B8 WwVvUu",

"Gänsefüßchen 01102qPp BÜöAäüÖ 8Bb66 Müllsäcke 11L000g6B8 WwVvUu",

"Gänsefüßchen 0l102qPp ßUöÄäÜÖ 8Bb66 Müllsäcke 11lL000g6B8 WwVvUu",

"Gänsefüßchen OL102qPp BÜöÄäÜÖ 8Bb66 Müllsäcke 11lL000g6B8 WwVvUu",

"Gänsefüßchen 01102qPp BüöÄäüÖ 8Bb66 Müllsäcke 11L000g96B8 WwVvUu",
]
#35
claude_form_res = read_and_normalize_file("Claude35_form.txt")

gemini2_responses = [
'Mülltrennung ist wichtig für die Umwels Ök01091scher Fußabdruck hängt von vielen Faktoren ab. "Üsergroßen passen nicht 1mm3r in d4s Standardregal Gänseßchen setzten ist manchmal kompliziert', 

'Mülltrennung ist wichtig für die Umwels Ökologischer Fußabdruck hängt von vielen Faktoren ab. "Übergrößen passen nicht immer in das Standardregal Gänseßchen setzten ist manchmal Kompliziert',

'Mülltrennung ist wichtig für die Umwels Ökologischer Fußabdruck hägt von vielen Faktoren ab. "Üsergroßen passen nicht 1mm3r in d4s Standardregal Gänseßchen setzten 1st manchmal Kompliziert',

'Mülltrennung ist wichtig für die Umwels Ökologischer Fußabdruck hägt von vielen Faktoren ab. "Üsergroßen passen nicht immer in das Standardregal Gänseßchen setzten 1st manchmal kompliziert',

'Mülltrennung ist wichtig für die Umwels Öko10g1scher Fußabdruck hägt von vielen Faktoren ab. "Üsergroßen passen nicht 1mm3r in d4s Standardregal Gänseßchen setzten ist manchmal Komp21z1ert',
]

gemini2_jargon_responses = [
"Gänsefüßchen α10QqPp BüÖÄäüÖ 8BbG6 Müllsäcke 114L00996B8 WwVvUu",

"Gänsefüßchen ol10QqPp BüoAäüÖ 8BbG6 Müllsäcke 111LOOog9G6B8 WwVvUu",

"Gänsefüßchen ol102qPp. BüöÄäüÖ 8BbG6 Müllsäcke 111L00996B8 WwVvUu",

"Gänsefüßchen Ol10QqPp BÜöÄäÜö 8BbG6 Müllsäcke 114L000996B8 WwVvUu",

"Gänsefüßchen Ol10Q9Pp BüöÄäüÖ 8BbG6 Müllsäcke 114L0O9g6B8 WwVvUu",
]

gemini2_form_res = read_and_normalize_file("Gemini2_form.txt")

gpt1pro_responses = [
"Mülltrennung 1st wichtig für die Umwel8 ÖKOl0g1scher Fußabdruck hängt von v1elen Fakt0ren ab. Übergroßen pass3n n1cht 1mm3r in d4s St4ndardregal Gänseßchen setzen 1st m4nchmal k0mpl1z1ert",

"Mülltrennung 1st wichtig für die Umwel8 Ök0l0g1scher Fußabdruck hängt von v1elen Faktoren ab. Ü8ergroßen pass3n n1cht 1mm3r in d4s St4ndardregal Gänseßchen setzen 1st m4nchmal k0mpl1ziert",

"Mülltrennung 1st wichtig für die Umwel8 Ök0l0g1scher Fußabdruck hängt von v1e1en Fakt0ren ab. Übergroßen passen n1cht 1mm3r in d4s St4ndardregal Gänseßchen setzen 1st m4nchmal k0mp1z1ert",

"Mülltrennung 1st wichtig für die Umwel8 ÖKOl0g1scher Fußabdruck hängt von v1e1en Fakt0ren ab. Übergroßen pass3n n1cht 1mm3r in d4s St4ndardregal Gänseßchen setzen 1st m4nchmal k0mpl1z1ert",

"Mülltrennung 1st wichtig für die Umwel8 ÖKO10g1scher Fußabdruck hängt von v1e1en Fakt0ren ab. Übergroßen pass3n n1cht 1mm3r in d4s St4ndardregal Gänseßchen setzen 1st m4nchmal k0mpl1z1ert",
]

gpt1pro_jargon_responses = [
"Gänsefüßchen @10aqp BlöÄäüÖ 8BbG6 Müllsäcke 111L00 9gG8 WwVvUu",

"Gänsefüßchen 0110QqPp BlöÄäüÖ 8BbG6 Müllsäcke 111L00gg6gB8 WwVvUu",

"Gänsefüßchen @010qQp BüöÄäüÖ 8BbG6 Müllsäcke 11L00 9g6B8 WwVvUu",

"Gänsefüßchen @10QqP BüöÄäüÖ 8BbG6 Müllsäcke 11L00 9gG8 WwVvUu",

"Gänsefüßchen @10aqp. BlüöÄäüÖ 8BbG6 Müllsäcke 111OO99g6B8 WwVvUu",
]

gpt1pro_form_res = read_and_normalize_file("o1_pro_form.txt")

deepseekV3_responses = [
"Multivennung ist wichtig für die Umwelt Ökologischer Fußabdruck. Hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal. Gemeinsam setzen ist manchmal kompliziert.",

"Multivennung ist wichtig für die Umwelt Ökologischer Fußabdruck. Magi von vielen Faktoren ab. Übergroßen passen nicht immer in das Standardregal. Gemeβchen setzten ist mündimät Kompliziert.",

"Multivennung ist wichtig für die Umwelt Ökologischer Fußabdruck. Hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal. Gemeinsam handeln ist manchmal kompliziert.",

"Multivennung ist wichtig für die Umwelt Ökologischer Fußabdruck. Hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal. Gemeinsam handeln ist manchmal kompliziert.",

"Multivennung ist wichtig für die Umwelt Ökologischer Fußabdruck. Hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal. Gemeinsam setzen ist manchmal kompliziert.",
]

deepseekV3_jargon_responses = [
"Ganschapchen 0100a9p. Richtlich 88b66 Multiscale 111100o3g688 www.u",

"Ganschapchen 0100a9p. Richtlich 88b66 Multiscale 111100o3g688 www.u",

"Ganschapchen 0100a9p. Richtlich 88b66 Multiscale 111100o3g688 www.u",

"Ganschapchen 0100a9p. Richtlich 88b66 Multiscale 111100o3g688 www.u",

"Ganschapchen 0100a9p. Richtlich 88b66 Multiscale 111100o3g688 www.u",
]

deepseekV3_form_res = read_and_normalize_file("Deepseek_form.txt")

gemini1_responses = [
"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergroßen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergroßen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergroßen passen nicht immer in das Standardregal Gänseßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergroßen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergroßen passen nicht immer in das Standardregal Gänseßchen setzen ist manchmal kompliziert",
]

gemini1_jargon_responses = [
"Gänsefüßchen 0100qp BüöÄüÖ 8Bb66 Müllsocke 11LOOggB8 WvVU",

"Gänsefüßchen 01,10QP BüÄüÖ 8Bß6 Müllsocke 114LOOßg6B8 WvVu",

"Gänsefüßchen 01.10 Qp. BüÄüÖ 8ßb6 Müllsocke 11/100 gg ß8 Wv Vu",

"Gänsefüßchen 01,10 Qp. BÜÄÜÖ 8ßþ6 Müllsocke 114L00.9g6ß8 WvVu",

"Gänsefünchen 0110QqP BüÄäÜÖ 8Bß66 Müllsocke 11200.9g6B8 WvVU",
]

gemini1_form_res = read_and_normalize_file("Gemini1_form.txt")

gpt4o_responses = [
"Mülltrennung 1st wichtig für die Umwel8 ÖkOlOq1scher Fußabdruck hängr von vieIen Faktoren ab. Übergrößen pass3n n1cht 1mm3r in das Standardregal Gänseßchen setzen 1st m4nchm4I kOmpIzi4rt",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergroßen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt. Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergroßen passen nicht immer in das Standardregal. Gänsefüßchen setzen ist manchmal kompliziert.",

"Mülltrennung 1st wichtig für die Umwel8 Ök0l0g1scher Fußabdruck hängt von v1elen Fakt0ren ab. Üßergrößen pass3n n1cht 1mm3r in d4s St4ndardregal Gänseßen setzten 1st m4nchmal k0mpl1z1ert",
]

gpt4o_jargon_responses = [
"Gänsefüßchen 010QqPp ßüöÄäüÖ 8BbG6 Müllsäcke 111L00ggG8 WwVvUu",

"Gänsefüßchen @10qQp. BlöÄäüÖ 8BbG6 Müllsäcke 11L00ggG8 WwVvUu",

"Gänsefűßchen Δ10qpP. ßLöÄäüÖ 8BbG6 Műllsäck 111L009gG8 wwVvUu",

"Gänsefußchen Ø10QqPp BÜÖÄäüÖ 8BbG6 Müllsacke 111L0O0gG6B8 WwVvUu",

"Gänsefübchen O1oqp BÜoÄäüÖ 8BbG6 Müllsäcke 11LOO9gG8 wvVvUu",
]

gpt4o_form_res = read_and_normalize_file("gpt4o_form.txt")

claude_37_responses = [
"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",

"Mülltrennung ist wichtig für die Umwelt Ökologischer Fußabdruck hängt von vielen Faktoren ab. Übergrößen passen nicht immer in das Standardregal Gänsefüßchen setzen ist manchmal kompliziert",
]

claude_37_jargon_responses = [
"Gänsefüßchen α110aqPp ßÜöÄäüÖ 8ßb66 Müllsäcke 111L00og6ß8 WwVvUu.",

"Gänsefüßchen α10aqPp ßÜöÄäüÖ 8ßbG6 Müllsäcke 111L0oQg6ß8 WwVvUu.",

"Gänsefüßchen α10αqPp ßÜöÄäüÖ 8ßbGG Müllsäcke 111L0oogg6ß8 WwVv Uu.",

"Gänsefüßchen α10aqPp ßÜöÄäüÖ 8ßbG6 Müllsäcke 111L0o0g6ß8 WwVvUu.",

"Gänsefüßchen α10aqPp ßÜöÄäüÖ 8ßbG6 Müllsäcke 111LOOog6ß8 WwVv Uu.",
]

claude_37_form_res = read_and_normalize_file("Claude37_form.txt")

tesseract_jargon = [
    '_";“"*?a.‚a\f"‚. ‚) %"ü*[ﬁi \ ia:.üi. M S .-""Gunst““ß°hé/h/‘f*’ﬁ0(4" » BU6AAC C6 P @%£é:ä&«ﬁ?%& W ?)éé;yo.m{l„or(«h/;fgmgaßk 66ü Müllsocke ‘ 114L0069a6B@ WulvUu . S 20 E - A R Y 4 4 +',

    '_";“"*?a.‚a\f"‚. ‚) %"ü*[ﬁi \ ia:.üi. M S .-""Gunst““ß°hé/h/‘f*’ﬁ0(4" » BU6AAC C6 P @%£é:ä&«ﬁ?%& W ?)éé;yo.m{l„or(«h/;fgmgaßk 66ü Müllsocke ‘ 114L0069a6B@ WulvUu . S 20 E - A R Y 4 4 +',

    '_";“"*?a.‚a\f"‚. ‚) %"ü*[ﬁi \ ia:.üi. M S .-""Gunst““ß°hé/h/‘f*’ﬁ0(4" » BU6AAC C6 P @%£é:ä&«ﬁ?%& W ?)éé;yo.m{l„or(«h/;fgmgaßk 66ü Müllsocke ‘ 114L0069a6B@ WulvUu . S 20 E - A R Y 4 4 +',

    '_";“"*?a.‚a\f"‚. ‚) %"ü*[ﬁi \ ia:.üi. M S .-""Gunst““ß°hé/h/‘f*’ﬁ0(4" » BU6AAC C6 P @%£é:ä&«ﬁ?%& W ?)éé;yo.m{l„or(«h/;fgmgaßk 66ü Müllsocke ‘ 114L0069a6B@ WulvUu . S 20 E - A R Y 4 4 +',

    '_";“"*?a.‚a\f"‚. ‚) %"ü*[ﬁi \ ia:.üi. M S .-""Gunst““ß°hé/h/‘f*’ﬁ0(4" » BU6AAC C6 P @%£é:ä&«ﬁ?%& W ?)éé;yo.m{l„or(«h/;fgmgaßk 66ü Müllsocke ‘ 114L0069a6B@ WulvUu . S 20 E - A R Y 4 4 + '
]

tesseract_leetspeak = [
    'S e A -—g“vt[“&%l4uﬂcg 4S w‘(cl„;%yw?j{gip};e (}„E E S ﬁj«@äﬂs‚g„@g&‚%%„‚ äQIQQ4O%’;E‚%Q% FußbAruck höar von v4£19‚\ ; S N oln äsﬁ;é@.% S :ﬁä ä\\ ; Eg; $"" A ä_‚-;f" :.E‚-_‚-=""" z SE E z;k;ﬁ.iä-%@‘iäe‘:_@u an UBeravoßen . pass3n nAcht Ammö3r in d4ts - 5 S - ) CC B E Gönseßchen setzsen Ast mandım&l KOmpl4z1ert —',

    'S e A -—g“vt[“&%l4uﬂcg 4S w‘(cl„;%yw?j{gip};e (}„E E S ﬁj«@äﬂs‚g„@g&‚%%„‚ äQIQQ4O%’;E‚%Q% FußbAruck höar von v4£19‚\ ; S N oln äsﬁ;é@.% S :ﬁä ä\\ ; Eg; $"" A ä_‚-;f" :.E‚-_‚-=""" z SE E z;k;ﬁ.iä-%@‘iäe‘:_@u an UBeravoßen . pass3n nAcht Ammö3r in d4ts - 5 S - ) CC B E Gönseßchen setzsen Ast mandım&l KOmpl4z1ert —',
    
    'S e A -—g“vt[“&%l4uﬂcg 4S w‘(cl„;%yw?j{gip};e (}„E E S ﬁj«@äﬂs‚g„@g&‚%%„‚ äQIQQ4O%’;E‚%Q% FußbAruck höar von v4£19‚\ ; S N oln äsﬁ;é@.% S :ﬁä ä\\ ; Eg; $"" A ä_‚-;f" :.E‚-_‚-=""" z SE E z;k;ﬁ.iä-%@‘iäe‘:_@u an UBeravoßen . pass3n nAcht Ammö3r in d4ts - 5 S - ) CC B E Gönseßchen setzsen Ast mandım&l KOmpl4z1ert —',
    
    'S e A -—g“vt[“&%l4uﬂcg 4S w‘(cl„;%yw?j{gip};e (}„E E S ﬁj«@äﬂs‚g„@g&‚%%„‚ äQIQQ4O%’;E‚%Q% FußbAruck höar von v4£19‚\ ; S N oln äsﬁ;é@.% S :ﬁä ä\\ ; Eg; $"" A ä_‚-;f" :.E‚-_‚-=""" z SE E z;k;ﬁ.iä-%@‘iäe‘:_@u an UBeravoßen . pass3n nAcht Ammö3r in d4ts - 5 S - ) CC B E Gönseßchen setzsen Ast mandım&l KOmpl4z1ert —',
    
    'S e A -—g“vt[“&%l4uﬂcg 4S w‘(cl„;%yw?j{gip};e (}„E E S ﬁj«@äﬂs‚g„@g&‚%%„‚ äQIQQ4O%’;E‚%Q% FußbAruck höar von v4£19‚\ ; S N oln äsﬁ;é@.% S :ﬁä ä\\ ; Eg; $"" A ä_‚-;f" :.E‚-_‚-=""" z SE E z;k;ﬁ.iä-%@‘iäe‘:_@u an UBeravoßen . pass3n nAcht Ammö3r in d4ts - 5 S - ) CC B E Gönseßchen setzsen Ast mandım&l KOmpl4z1ert —'
]

tesseract_form_res = read_and_normalize_file("tesseract_form.txt","#")


#this for convience since i dont have time to make it pretty but the exact same values can be found on the folder wrong_text/modelName
#confvalues
gpt4oLeetspeak_Confvalues = [
    [95,99,95,95,95,90,85,95,90,99,95,95,99,95,90,90,90,90,99,99,95,90,95,90,85],
    [95,95,95,90,95,95,85,90,90,95,95,95,95,90,85,90,85,95,95,95,90,95,95,95,95],
    [95,100,100,100,100,95,90,95,100,100,100,100,100,95,100,100,95,100,100,100,100,100,100,100,100],
    [100,100,100,100,100,100,90,100,100,100,100,100,100,90,100,100,100,100,100,100,90,100,100,100,100],
    [95,90,95,95,95,90,85,95,95,95,90,90,95,90,90,90,90,95,90,90,90,95,90,90,85],
]

gpto1Leetspeak_Confvalues = [
    [99,98,98,99,99,90,92,99,99,99,94,93,99,99,95,95,94,99,93,88,90,99,98,94,90],
    [96,95,97,96,98,92,90,96,95,97,93,97,97,89,92,93,92,99,91,90,93,98,95,91,89],
    [98,97,98,98,99,91,91,98,97,98,94,95,98,94,94,94,93,99,92,90,92,99,97,93,90],
    [95,80,95,95,95,85,80,95,95,95,80,85,95,90,80,85,80,95,85,85,90,95,80,85,85],
    [97,93,97,97,98,89,88,97,97,97,93,93,97,93,91,92,91,98,93,88,91,98,94,91,88],
]
#Functions

def similarity_percentage(str1: str, str2: str) -> float:
    # Compute Levenshtein distance
    distance = ds(str1, str2)
    # Compute similarity percentage
    similarity = max(0,round((1 - distance / len(str1)) * 100, 2))
    return similarity

def standard_deviation(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0  # Avoid division by zero

    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    return round(math.sqrt(variance), 2)

def normalize_text(text: str) -> str:
    return " ".join(text.split())  # Removes extra spaces and joins lines

def printStat(model: str, type: str, acc, avg, sd):
    print(model)
    print(type)
    print("ACC:",acc, "AVG:",avg, "SD:",sd)
    print("----------------------")

#make it easier to use properties for printstat
class Result:
    def __init__(self, acc, avg, sd):
        self.acc = acc
        self.avg = avg
        self.sd = sd

#just to make it easier to write code with autocompletions and type safety
class CM_Result:
    def __init__(self, precicion, accuracy, sensitivity, f1):
        self.precision = precicion
        self.accuracy = accuracy
        self.sensitivity = sensitivity
        self.f1 = f1

class for_ROC_Result:
    def __init__(self, auc, tpr, fpr):
        self.auc = auc
        self.tpr = tpr
        self.fpr = fpr
        
class cross_Val_Result:
    def __init__(self, precisions, mean, std):
        self.precisions = precisions
        self.mean = mean
        self.std = std

def calc(model:str, type: str, gt: str, res):
    acc = [similarity_percentage(gt, response) for response in res]
    avg = round(sum(acc) / len(acc), 2)
    sd = standard_deviation(acc)
    printStat(model, type, acc, avg, sd)
    return Result(acc, avg, sd)

def visual(data, title: str):
    # X-axis values
    x = np.arange(1, 6)
    #markers = ['o', 'x', '*', 'X', '□']
    # Plot each array as a line
    plt.figure(figsize=(8, 5))
    for label, values in data.items():
        plt.plot(x, values, marker='o', linestyle='-', label=label)

    # Labels and title
    plt.xlabel("Versuch")
    plt.ylabel("Ähnlichkeit in Prozent")
    plt.title(title)
    plt.subplots_adjust(bottom=0.2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=8, frameon=False) # Move legend outside
    plt.xticks(np.arange(1, 6, 1))  # Set x-axis increments to 1
    plt.grid(True)

    # Show the plot
    plt.show()

def visualROC(data, title: str):
    #markers = ['o', 'x', '*', 'X', '□']
    # Plot each array as a line
    plt.figure(figsize=(8, 5))
    i = 0
    for label, values in data.items():
        plt.plot(values.fpr, values.tpr, marker='', linestyle='-', label=label)
        text_box = f"{label} AUC: {values.auc:.3f}"
        plt.text(0.95, 0.95 - (i * 0.05), text_box, transform=plt.gca().transAxes,
                 fontsize=8, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.5))
        i = i +1
    
    # Plot the random classifier line
    plt.plot([0, 1], [0, 1], linestyle='--')
    # Labels and title
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.subplots_adjust(bottom=0.2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=8, frameon=False) # Move legend 
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_bar_graph(modes, categories, values):
    num_modes = len(modes)
    num_categories = len(categories)
    
    x = np.arange(num_modes)  # Position for each mode
    width = 0.2  # Width of each bar
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(num_categories):
        ax.bar(x + i * width, [values[j][i] for j in range(num_modes)], width, label=categories[i])
    
    ax.set_xticks(x + width * (num_categories / 2 - 0.5))
    ax.set_xticklabels(modes)
    ax.set_ylabel("Ähnlichkeit in Prozent")
    ax.set_title("Durschnittliche Ähnlichkeit")
    plt.grid(True)
    ax.legend()
    
    plt.show()

def plot_model_stddevs(values, model_labels, category_labels):
    models = len(values)
    categories = len(values[0])
    x = np.arange(models)
    width = 0.2  # Width of the bars
    
    plt.figure(figsize=(12, 6))
    for i in range(categories):
        category_values = [values[m][i] for m in range(models)]
        plt.bar(x + i * width, category_values, width, label=category_labels[i])
    
    plt.xticks(x + width * (categories / 2 - 0.5), model_labels, rotation=45)
    plt.ylabel("Standard Abweichung")
    plt.title("Standard Abweichung von verschiedene Modellen")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def read_csv(path:str):
    results = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)
    return results

def align_responses(ground_truth, recognized_words):
    # Split the ground truth into words
    ground_truth_words_padded = ground_truth.split()

    # split resp
    response_words_padded = recognized_words.split()

    #find length
    len1, len2 = len(ground_truth_words_padded), len(response_words_padded)

    if len1 > len2:
        response_words_padded.extend([""] * (len1 - len2))
    elif len2 > len1:
        ground_truth_words_padded.extend([""] * (len2 - len1))

    return ground_truth_words_padded, response_words_padded

def binary_classification(ground_truth, recognized_words):
    # Align the responses and the ground truth if needed
    ground_truth_words_padded, response_words_padded = align_responses(ground_truth, recognized_words)

    # Prepare the result for y_true
    y_true = []
    
    # Compare each word index
    for i in range(len(ground_truth_words_padded)):
        if(ground_truth_words_padded[i] == response_words_padded[i]):
            y_true.append(1)
        else:
            y_true.append(0)

    return y_true

def get_roc_and_pr(binary_cl, conf_value):
    #roc expects values from 0 to 1 nto 0 to 100
    avg_scores = np.array(conf_value) / 100
    fpr, tpr, _ = roc_curve(binary_cl, avg_scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

def roc_metrics(data, gt):
    answers = {}
    for label, values in data.items():
        print(label)
        res = average_roc(gt, values)
        answers[label] = res
    visualROC(answers, "ROC")

def roc_conf_values(gt, data):
    answers = {}
    for label, values in data.items():
        aucs = []
        fprs = []
        tprs = []
        i = 0
        for response in values["res"]:
            ytrue = binary_classification(gt, response)
            roc_auc, fpr, tpr = get_roc_and_pr(ytrue, values["conf"][i])
            aucs.append(roc_auc)
            fprs.append(fpr)
            tprs.append(tpr)
        res = for_ROC_Result(np.mean(aucs), np.mean(tprs, axis=0), np.mean(fprs, axis=0))
        answers[label] = res
    visualROC(answers, "ROC")

def create_metric_table(answers):

    labels = list(answers.keys())
    metrics = ["Precision", "Accuracy", "Sensitivity", "F1"]
    data = []

    for metric in metrics:
        row = []
        for label in labels:
            result = answers[label]
            if metric == "Precision":
                row.append(result.precision)
            elif metric == "Accuracy":
                row.append(result.accuracy)
            elif metric == "Sensitivity":
                row.append(result.sensitivity)
            elif metric == "F1":
                row.append(result.f1)
        data.append(row)

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=labels, rowLabels=metrics, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.5) #adjust table size

    plt.show()

def create_cross_table(answers, title):

    labels = list(answers.keys())
    metrics = ["Durch. Genauigkeit", "Standardabweichung"]
    data = []

    for metric in metrics:
        row = []
        for label in labels:
            result = answers[label]
            if metric == "Durch. Genauigkeit":
                row.append(result.mean)
            elif metric == "Standardabweichung":
                row.append(result.std)
        data.append(row)

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=labels, rowLabels=metrics, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.5) #adjust table size

    plt.show()

def ensure_roc_endpoints(fpr, tpr):

    has_zero_zero = False
    has_one_one = False

    #for i in range(len(fpr)):
    #    if fpr[i] == 0.0 and tpr[i] == 0.0:
    #        has_zero_zero = True
    #    if fpr[i] == 1.0 and tpr[i] == 1.0:
    #        has_one_one = True
#
    #if not has_zero_zero:
    #    fpr.insert(0, 0.0)
    #    tpr.insert(0, 0.0)
#
    #if not has_one_one:
    #    fpr.append(1.0)
    #    tpr.append(1.0)

    # Sort based on FPR
    zipped_lists = zip(fpr, tpr)
    sorted_pairs = sorted(zipped_lists)

    sorted_fpr, sorted_tpr = zip(*sorted_pairs)

    return list(sorted_fpr), list(sorted_tpr)

def calculate_auc(fpr, tpr):
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return auc

def calculate_tp_fp_fn_tn(ground_truth, responses, modelName):
    gt_chars = list(ground_truth)  # Convert to list for position-based comparison
    pl_answers =  []
    for i, response in enumerate(responses):
        response_chars = list(response)

        tp = sum(1 for x, y in zip(gt_chars, response_chars) if x == y and x != " ") #correct chars
        tn = sum(1 for x, y in zip(gt_chars, response_chars) if x == y and x == " ") #correct whitespace
        fp = max(0, len(response_chars) - tp - tn)  # Extra characters in response
        fn = max(0, len(gt_chars) - tp - tn)  # Missing characters from ground truth
        
        print(f"{modelName}: {i+1}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

        precicion = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        #in case of tesseract where the results are so bad that precision and sens are zero, it would throw a zero division error so just patch to 0 by default
        #mathematically it should be fine
        f1 =  2 * ((precicion * sensitivity) / (precicion + sensitivity)) if precicion + sensitivity != 0 else 0.0
        pl_answers.append(CM_Result(precicion,accuracy,sensitivity,f1))
    
    precisions = [result.precision for result in pl_answers]
    accuracies = [result.accuracy for result in pl_answers]
    sensitivities = [result.sensitivity for result in pl_answers]
    f1_scores = [result.f1 for result in pl_answers]

    avg_precision = sum(precisions) / len(precisions)
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_sensitivity = sum(sensitivities) / len(sensitivities)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    
    #round only at the return to keep calcualtions as accurate as possible
    return CM_Result(round(avg_precision, 2), round(avg_accuracy, 2), round(avg_sensitivity, 2), round(avg_f1, 2))

def calculate_auc_roc(ground_truth, responses, modelName):
    gt_chars = list(ground_truth)  # Convert to list for position-based comparison
    fpr = []
    tpr = []
    for i, response in enumerate(responses):
        response_chars = list(response)

        tp = sum(1 for x, y in zip(gt_chars, response_chars) if x == y and x != " ") #correct chars
        tn = sum(1 for x, y in zip(gt_chars, response_chars) if x == y and x == " ") #correct whitespace
        fp = max(0, len(response_chars) - tp - tn)  # Extra characters in response
        fn = max(0, len(gt_chars) - tp - tn)  # Missing characters from ground truth
        #print(f"{modelName}: {i+1}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        precicion = tp / (tp + fp)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        tpr.append(precicion)

    fpr2, tpr2 = ensure_roc_endpoints(fpr, tpr)
    aucVal = calculate_auc(fpr2, tpr2)
    print(modelName,": AUC: ", aucVal)

    return for_ROC_Result(aucVal, tpr2, fpr2)
    
    
    #round only at the return to keep calcualtions as accurate as possible

def metrics(data, title, gt):
    answers = {}
    for label, values in data.items():
        res = calculate_tp_fp_fn_tn(gt, values, label)
        answers[label] = res
    create_metric_table(answers)    
    #print(answers.keys, answers.values)

def metricsROC(data, gt, title):
    answers = {}
    for label, values in data.items():
        res = calculate_auc_roc(gt, values, label)
        answers[label] = res
    visualROC(answers, "test")

    #create_metric_table(answers)    
#simplification is allowed for this usecase so i can get the accuracy directly
#mathematicallz it should be ok
def character_accuracy(ground_truth, response):
    correct = 0
    total = min(len(ground_truth), len(response))
    for i in range(total):
        if ground_truth[i] == response[i]:
            correct += 1
    return correct / total if total > 0 else 0

def calculate_k_cross(gt, responses):
    ground_truth_chars = list(gt)
    responses_chars = [list(response) for response in responses]

    k = len(responses_chars)
    accuracies = []

    for i in range(k):
        test_response = responses_chars[i]
        accuracy = character_accuracy(ground_truth_chars, test_response)
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"Durchschnittliche Genauigkeit: {average_accuracy}")
    print(f"Standardabweichung der Genauigkeit: {std_accuracy}")
    return accuracies

def metricsKCross(data, gt, title):
    answers = {}
    for label, values in data.items():
        res = calculate_k_cross(gt, values)
        answers[label] = cross_Val_Result(res, round(np.mean(res),2), round(np.std(res),2))
    create_cross_table(answers, title)

    #create_metric_table(answers)    

def word_level_levenshtein_roc_word_only(gt, response, threshold=1):

    ground_words = gt.split()
    response_words = response.split()

    y_true_all = []
    y_scores_all = []

    ground_index = 0
    response_index = 0

    while ground_index < len(ground_words) and response_index < len(response_words):
        ground_word = ground_words[ground_index]
        response_word = response_words[response_index]

        distance = ds(ground_word, response_word)
        
        
        y_true_all.append(1 if distance <= threshold else 0)
        y_scores_all.append(1 - (distance / max(len(ground_word), len(response_word), 1)))

        ground_index += 1
        response_index += 1

    while ground_index < len(ground_words):
        y_true_all.append(0)
        y_scores_all.append(0)
        ground_index +=1

    while response_index < len(response_words):
        y_true_all.append(0)
        y_scores_all.append(0)
        response_index +=1

    fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
    roc_auc = auc(fpr, tpr)
    #print(roc_auc)
    return fpr, tpr, roc_auc

def average_roc(gt, responses, threshold=1):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)  # Common FPR values

    for response in responses:
        fpr, tpr, roc_auc = word_level_levenshtein_roc_word_only(gt, response, threshold)
        if not np.isnan(roc_auc):
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0  # Ensure the starting point is 0
            aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0  # Ensure the ending point is 1
    mean_auc = auc(mean_fpr, mean_tpr)
    #print(mean_auc, mean_tpr, mean_fpr)
    return for_ROC_Result(mean_auc, mean_tpr, mean_fpr)

def roc_metrics(data, gt):
    answers = {}
    for label, values in data.items():
        print(label)
        res = average_roc(gt, values)
        answers[label] = res
    visualROC(answers, "ROC")

# Define results
#CLAUDE
claude35_wt = calc("Claude 3.5", "Wrong Text", ground_truth, claude_responses)
claude35_j = calc("Claude 3.5", "Jargon", ground_truth_jargon, claude_jargon_response)
claude35_form = calc("Claude 3.5", "Form", ground_truth_form, claude_form_res)
#GEMINI2
gemini2_wt = calc("Gemini 2", "Wrong Text", ground_truth, gemini2_responses)
gemini2_j = calc("Gemini 2", "Jargon", ground_truth_jargon, gemini2_jargon_responses)
gemini2_form = calc("Gemini 2", "Form", ground_truth_form, gemini2_form_res)
#GPT o1pro
gpt1pro_wt = calc("GPT-1pro", "Wrong Text", ground_truth, gpt1pro_responses)
gpt1pro_j = calc("GPT-1pro", "Jargon", ground_truth_jargon, gpt1pro_jargon_responses)
gpt1pro_form = calc("GPT-1pro", "Form", ground_truth_form, gpt1pro_form_res)
#Deepseek V3
deepseekV3_wt = calc("DeepSeek V3", "Wrong Text", ground_truth, deepseekV3_responses)
deepseekV3_j = calc("DeepSeek V3", "Jargon", ground_truth_jargon, deepseekV3_jargon_responses)
deepseekV3_form = calc("DeepSeekV3", "Form", ground_truth_form, deepseekV3_form_res)
#GEMINI1
gemini1_wt = calc("Gemini 1.5", "Wrong Text", ground_truth, gemini1_responses)
gemini1_j = calc("Gemini 1.5", "Jargon", ground_truth_jargon, gemini1_jargon_responses)
gemini1_form = calc("Gemini 1.5", "Form", ground_truth_form, gemini1_form_res)
#GPT 4o
gpt4o_wt = calc("GPT 4o", "Wrong Text", ground_truth, gpt4o_responses)
gpt4o_j = calc("GPT 4o", "Jargon", ground_truth_jargon, gpt4o_jargon_responses)
gpt4o_form = calc("GPT-1pro", "Form", ground_truth_form, gpt4o_form_res)
#CLAUDE 37
claude37_wt = calc("Claude 3.7", "Wrong Text", ground_truth, claude_37_responses)
claude37_j = calc("Claude 3.7", "Jargon", ground_truth_jargon, claude_37_jargon_responses)
claude37_form = calc("Claude 3.7", "Form", ground_truth_form, claude_37_form_res)

#teseract
tess_wt = calc("Teeseract", "Wrong Text", ground_truth, tesseract_leetspeak)
tess_j = calc("Teeseract", "Jargon", ground_truth_jargon, tesseract_jargon)
tess_form = calc("Tesseract", "Form", ground_truth_form, tesseract_form_res)

letter_wrong_text = {
    "Claude 3.5": claude35_wt.acc,
    "Claude 3.7": claude37_wt.acc,
    "Gemini 1.5": gemini1_wt.acc,
    "Gemini 2.0": gemini2_wt.acc,
    "GPT 4o": gpt4o_wt.acc,
    "GPT o1 pro": gpt1pro_wt.acc,
    "Deepseek V3": deepseekV3_wt.acc,
    "Tesseract": tess_wt.acc
}

letter_jargon = {
    "Claude 3.5": claude35_j.acc,
    "Claude 3.7": claude37_j.acc,
    "Gemini 1.5": gemini1_j.acc,
    "Gemini 2.0": gemini2_j.acc,
    "GPT 4o": gpt4o_j.acc,
    "GPT o1 pro": gpt1pro_j.acc,
    "Deepseek V3": deepseekV3_j.acc,
    "Tesseract": tess_j.acc
}

letter_form = {
    "Claude 3.5": claude35_form.acc,
    "Claude 3.7": claude37_form.acc,
    "Gemini 1.5": gemini1_form.acc,
    "Gemini 2.0": gemini2_form.acc,
    "GPT 4o": gpt4o_form.acc,
    "GPT o1 pro": gpt1pro_form.acc,
    "Deepseek V3": deepseekV3_form.acc,
    "Tesseract": tess_form.acc
}

models = ["Claude 3.5", "Claude 3.7", "Gemini 1.5", "Gemini 2.0", "GPT 4o", "GPT o1 pro", "DeepSeek V3", "Tesseract"]
categories = ["Leetspeak", "Jargon", "Bachelorantrag"]
averages = [
    [claude35_wt.avg, claude35_j.avg, claude35_form.avg],
    [claude37_wt.avg, claude37_j.avg, claude37_form.avg],
    [gemini1_wt.avg, gemini1_j.avg, gemini1_form.avg],
    [gemini2_wt.avg, gemini2_j.avg, gemini2_form.avg],
    [gpt4o_wt.avg, gpt4o_j.avg, gpt4o_form.avg],
    [gpt1pro_wt.avg, gpt1pro_j.avg, gpt1pro_form.avg],
    [deepseekV3_wt.avg, deepseekV3_j.avg, deepseekV3_form.avg],
    [tess_wt.avg, tess_j.avg, tess_form.avg]
]
sd = [
    [claude35_wt.sd, claude35_j.sd, claude35_form.sd],
    [claude37_wt.sd, claude37_j.sd, claude37_form.sd],
    [gemini1_wt.sd, gemini1_j.sd, gemini1_form.sd],
    [gemini2_wt.sd, gemini2_j.sd, gemini2_form.sd],
    [gpt4o_wt.sd, gpt4o_j.sd, gpt4o_form.sd],
    [gpt1pro_wt.sd, gpt1pro_j.sd, gpt1pro_form.sd],
    [deepseekV3_wt.sd, deepseekV3_j.sd, deepseekV3_form.sd],
    [tess_wt.sd, tess_j.sd, tess_form.sd]
]

leetspeak_res = {
    "Claude 3.5": claude_responses,
    "Claude 3.7": claude_37_responses,
    "Gemini 1.5": gemini1_responses,
    "Gemini 2.0": gemini2_responses,
    "GPT 4o": gpt4o_responses,
    "GPT o1 pro": gpt1pro_responses,
    "Deepseek V3": deepseekV3_responses,
    #"Tesseract": tesseract_leetspeak
}

leetspeak_res_conf = {
    "GPT 4o": {"res" : gpt4o_responses, "conf": gpt4oLeetspeak_Confvalues} ,
    "GPT o1 pro": {"res" : gpt1pro_responses, "conf": gpto1Leetspeak_Confvalues},
}
jargon_res = {
    "Claude 3.5": claude_jargon_response,
    "Claude 3.7": claude_37_jargon_responses,
    "Gemini 1.5": gemini1_jargon_responses,
    "Gemini 2.0": gemini2_jargon_responses,
    "GPT 4o": gpt4o_jargon_responses,
    "GPT o1 pro": gpt1pro_jargon_responses,
    "Deepseek V3": deepseekV3_jargon_responses,
    "Tesseract": tesseract_jargon
}

form_res = {
    "Claude 3.5": claude_form_res,
    "Claude 3.7": claude_37_form_res,
    "Gemini 1.5": gemini1_form_res,
    "Gemini 2.0": gemini2_form_res,
    "GPT 4o": gpt4o_form_res,
    "GPT o1 pro": gpt1pro_form_res,
    "Deepseek V3": deepseekV3_form_res,
    "Tesseract": tesseract_form_res
}

#uncomment depending on what you want otherwise you will get 20 diagrams at once
#individual grpahs for performance
#visual(letter_jargon, "Jargon")
#visual(word_jargon, "Jargon per Word")
#visual(letter_form, "Bachelorantrag")

#grpahs for avg and sd
#plot_bar_graph(models, categories, averages)
#plot_model_stddevs(sd, models, categories)

#table for f1, acc, pre, sens 
#metrics(leetspeak_res, "Leetspeak", ground_truth)
#metrics(jargon_res, "Jargon", ground_truth_jargon)
#metrics(form_res, "Bachelorantrag", ground_truth_form)

#metricsROC(leetspeak_res, ground_truth, "Leetspeak")

# Cross Validation
#metricsKCross(leetspeak_res, ground_truth, "Cross Validation Leetspeak")
#metricsKCross(jargon_res, ground_truth_jargon, "Cross Validation Jargon")
#metricsKCross(form_res, ground_truth_form, "Cross Validation Form")

#average_roc(ground_truth, gemini1_responses)
#roc_metrics(leetspeak_res, ground_truth)
#roc_metrics(jargon_res, ground_truth_jargon)
#roc_metrics(form_res, ground_truth_form)

#roc_conf_values(ground_truth, leetspeak_res_conf)
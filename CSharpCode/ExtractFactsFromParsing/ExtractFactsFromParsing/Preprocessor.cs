using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class Preprocessor
{
    List<string> prefixes;
    List<string> infixes;

    public Preprocessor()
    {
        string[] sx =
        {
         "и|", // И я читал.
         "ну|и|", // Ну и жара нынче стоит!
         "к счастью|,", // К счастью, подавляющее большинство спасают.
         "в|итоге|,", // В итоге, кошка пропала.
         "а|вот", // А вот риф немного разочаровал.
         "вот|", // Вот так началась наша поездка!
         "ну|", // Ну ювелир принимает заказ.
         "а|", // А божественное право давало Слово.
         "конечно|", // Конечно, Мейми слегка растерялась.
         "но|", // Но куда же делись деньги ?
         "наконец|,", // Наконец, вверху помещается детектор.
         "иными|словами|,", // Иными словами, институт перестраховался.
         "короче|,", // Короче, предстоит переговорный процесс.
         "увы|,", // Увы, поезд стоял в...
         "во-вторых|,", // Во-вторых, удар смягчила вода.
         "по крайней мере|,", // По крайней мере, есть обнадеживающие факты.
         "в|общем|,", // В общем, набрали 4000 гривен.
         "да|и", // Да и обязанностями родители не отягощали.
         "хотя", // Хотя судьба Феликса сложилась трагически.
         "то есть", // То есть морская капуста попросту пропадала.
         "как|всегда|,", // Как всегда, не сходились цифры.
         "а|ведь", // А ведь поэты не шутят.
         "мол|", // Мол, суд потом разберется.
         "возможно|", // Возможно, правительство экономит деньги?
         "разумеется|", // Разумеется, поначалу храм блистал.
         "похоже|,|что|",
         "похоже|,|", // Похоже, разговор шел по-немецки.
        };

        prefixes = sx.Select(z => z.Split("|".ToCharArray(), StringSplitOptions.RemoveEmptyEntries)).OrderByDescending(z => z.Length).Select(z => string.Join("|", z) + "|").ToList();

        infixes = "же ли бы б ль ж".Split(' ').ToList();

    }



    public string Preprocess(string phrase0, SolarixGrammarEngineNET.GrammarEngine2 gren)
    {
        string phrase = phrase0;

        if (phrase.EndsWith(".."))
        {
            phrase = phrase.Substring(0, phrase.Length - 2);
        }

        if (phrase.EndsWith("!"))
        {
            phrase = phrase.Substring(0, phrase.Length - 1);
        }


        string[] tokens = gren.Tokenize(phrase, SolarixGrammarEngineNET.GrammarEngineAPI.RUSSIAN_LANGUAGE);
        List<string> res_tokens = tokens.ToList();
        bool changed = false;

        string s = string.Join("|", tokens).ToLower();

        foreach (string prefix in prefixes)
        {
            if (s.StartsWith(prefix))
            {
                // Ну и жара нынче стоит!
                res_tokens = res_tokens.Skip(prefix.Split("|".ToCharArray(), StringSplitOptions.RemoveEmptyEntries).Length).ToList();
                changed = true;
                break;
            }
        }

        foreach (string infix in infixes)
        {
            if (res_tokens.Contains(infix))
            {
                res_tokens.Remove(infix);
                changed = true;
            }
        }


        if (changed)
        {
            return string.Join(" ", res_tokens);
        }
        else
        {
            return phrase;
        }
    }

}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Security.Cryptography;
using System.Diagnostics.Contracts;

public class FootPrintToken
{
    private string word;
    private List<string> tags;
    private SolarixGrammarEngineNET.SyntaxTreeNode node;

    private static List<string> copula_verbs = "быть стать считаться оказаться получиться бывать становиться".Split().ToList();

    public FootPrintToken(string word)
    {
        this.word = word;
        tags = new List<string>();
        tags.Add(word);
    }

    public FootPrintToken(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode root)
    {
        Contract.Ensures(!string.IsNullOrEmpty(this.word));
        Contract.Ensures(this.node != null);
        Contract.Ensures(this.tags != null);

        this.word = root.GetWord();
        this.tags = new List<string>();
        this.node = root;

        this.tags.Add(root.GetWord().ToLower());

        if (root.GetWord().Equals("не", StringComparison.OrdinalIgnoreCase))
        {
            this.tags.Add("neg");
        }


        int part_of_speech = gren.GetEntryClass(root.GetEntryID());
        switch (part_of_speech)
        {
            case SolarixGrammarEngineNET.GrammarEngineAPI.NUM_WORD_CLASS: this.tags.Add("num"); break; // числительное цифрами
            case SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_CLASS_ru: this.tags.Add("num"); break; // числительное словом

            case SolarixGrammarEngineNET.GrammarEngineAPI.CONJ_ru: this.tags.Add("conj"); break; // союз

            case SolarixGrammarEngineNET.GrammarEngineAPI.PRONOUN_ru: this.tags.Add("pr"); break; // местоимение Я

            case SolarixGrammarEngineNET.GrammarEngineAPI.NOUN_ru: this.tags.Add("n"); break;

            case SolarixGrammarEngineNET.GrammarEngineAPI.ADJ_ru: this.tags.Add("adj"); break;
            case SolarixGrammarEngineNET.GrammarEngineAPI.VERB_ru: this.tags.Add("v"); break;
            case SolarixGrammarEngineNET.GrammarEngineAPI.INFINITIVE_ru: this.tags.Add("v"); break;
            case SolarixGrammarEngineNET.GrammarEngineAPI.GERUND_2_ru: this.tags.AddRange("adv adv_v".Split(' ')); break;

            case SolarixGrammarEngineNET.GrammarEngineAPI.ADVERB_ru:
                {
                    this.tags.Add("adv");
                    if (StringExtender.InCI(word, "очень крайне наиболее наименее чрезвычайно почти".Split())) // модификаторы наречий и прилагательных
                    {
                        this.tags.Add("a_modif");
                    }

                    string adv_cat = AdverbCategory.GetQuestionWordForAdverb(word);
                    if (!string.IsNullOrEmpty(adv_cat))
                    {
                        this.tags.Add("adv_" + adv_cat);
                    }

                    break;
                }

            case SolarixGrammarEngineNET.GrammarEngineAPI.PREPOS_ru: this.tags.Add("p"); break;
            case SolarixGrammarEngineNET.GrammarEngineAPI.PRONOUN2_ru: this.tags.Add("pr"); break;
            default: this.tags.Add("x"); break;
        }

        foreach (var p in root.GetPairs())
        {
            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.CASE_ru)
            {
                switch (p.StateID)
                {
                    case SolarixGrammarEngineNET.GrammarEngineAPI.NOMINATIVE_CASE_ru: this.tags.Add("nom"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.GENITIVE_CASE_ru: this.tags.Add("gen"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.ACCUSATIVE_CASE_ru: this.tags.Add("acc"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.DATIVE_CASE_ru: this.tags.Add("dat"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.PREPOSITIVE_CASE_ru: this.tags.Add("prep"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.PARTITIVE_CASE_ru: this.tags.Add("part"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.LOCATIVE_CASE_ru: this.tags.Add("loc"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.INSTRUMENTAL_CASE_ru: this.tags.Add("instr"); break;
                }
            }

            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru)
            {
                switch (p.StateID)
                {
                    case SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru: this.tags.Add("sing"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.PLURAL_NUMBER_ru: this.tags.Add("pl"); break;
                }
            }

            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.TENSE_ru)
            {
                switch (p.StateID)
                {
                    case SolarixGrammarEngineNET.GrammarEngineAPI.PAST_ru: this.tags.Add("past"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.PRESENT_ru: this.tags.Add("pres"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.FUTURE_ru: this.tags.Add("future"); break;
                }
            }

            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.FORM_ru)
            {
                switch (p.StateID)
                {
                    case SolarixGrammarEngineNET.GrammarEngineAPI.ANIMATIVE_FORM_ru: this.tags.Add("anim"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.INANIMATIVE_FORM_ru: this.tags.Add("inanim"); break;
                }
            }

            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.GENDER_ru)
            {
                switch (p.StateID)
                {
                    case SolarixGrammarEngineNET.GrammarEngineAPI.MASCULINE_GENDER_ru: this.tags.Add("masc"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.FEMININE_GENDER_ru: this.tags.Add("fem"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.NEUTRAL_GENDER_ru: this.tags.Add("neut"); break;
                }
            }


            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru)
            {
                switch (p.StateID)
                {
                    case SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_1_ru: this.tags.Add("1"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_2_ru: this.tags.Add("2"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_3_ru: this.tags.Add("3"); break;
                }
            }


            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.VERB_FORM_ru)
            {
                switch (p.StateID)
                {
                    case SolarixGrammarEngineNET.GrammarEngineAPI.VB_INF_ru: this.tags.Add("vf1"); break;
                    case SolarixGrammarEngineNET.GrammarEngineAPI.VB_ORDER_ru: this.tags.Add("imper"); break;
                }
            }

        }

        // Пометим связочные глаголы
        string lemma = gren.GetEntryName(root.GetEntryID());
        if (copula_verbs.Contains(lemma))
        {
            this.tags.Add("copula");
        }

    }

    public bool Match(string tags)
    {
        string[] tag_list = tags.Split(",.".ToCharArray());
        foreach (string t in tag_list)
        {
            if (t.StartsWith("~"))
            {
                if (this.tags.Contains(t.Substring(1, t.Length - 1)))
                {
                    return false;
                }
            }
            else if (t.Contains("|"))
            {
                string[] tx = t.Split('|');
                foreach (var ti in tx)
                {
                    if (this.tags.Contains(ti))
                    {
                        return true;
                    }
                }

                return false;
            }
            else
            {
                if (!this.tags.Contains(t))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public override string ToString()
    {
        return word + " (" + string.Join(",", tags) + ")";
    }

    public string GetWord() { return word; }
}


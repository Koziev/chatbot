using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Security.Cryptography;
using System.Diagnostics.Contracts;

class GenerateQAFromParsing
{
    static HashSet<string> stop_adjs;

    static Preprocessor preprocessor;

    static GenerateQAFromParsing()
    {
        stop_adjs = new HashSet<string>("один одна одно одни сам сама само сами".Split());
        preprocessor = new Preprocessor();
    }

    private static int sample_count = 0;

    static System.IO.StreamWriter wrt_samples, wrt_skipped;

    static int nb_samples = 0;
    private static void WriteQA(string premise, string question, string answer)
    {
        Contract.Requires(!string.IsNullOrEmpty(premise));
        Contract.Requires(!string.IsNullOrEmpty(question));
        Contract.Requires(!string.IsNullOrEmpty(answer));
        Contract.Requires(question != answer);
        Contract.Requires(premise != answer);

        wrt_samples.WriteLine();
        wrt_samples.WriteLine("T: {0}", premise.Trim());
        wrt_samples.WriteLine("Q: {0}", question.Trim());
        wrt_samples.WriteLine("A: {0}", answer.Trim());
        nb_samples += 1;
        return;
    }


    private static void WritePermutationsQA2(string premise, string answer, params string[] constituents)
    {
        if (constituents.Count(z => string.IsNullOrEmpty(z)) > 0)
        {
            throw new ArgumentNullException($"Null constituent in WritePermutationsQA2 call: premise={premise} answer={answer}");
        }


        int n = constituents.Length;
        int[] idx = new int[n];
        Array.Clear(idx, 0, n);

        string[] wx = new string[n];
        HashSet<string> generated = new HashSet<string>();

        int total_n = (int)Math.Round(Math.Pow(n, n));
        for (int p = 0; p < total_n; ++p)
        {
            bool good = true;
            for (int i = 0; i < n; ++i)
            {
                string c = constituents[idx[i]];
                for (int j = 0; j < i; ++j)
                {
                    if (wx[j] == c)
                    {
                        good = false;
                        break;
                    }
                }

                wx[i] = c;
            }

            if (good)
            {
                string sample = Join(wx) + "?";
                if (!generated.Contains(sample))
                {
                    generated.Add(sample);
                    WriteQA(premise, sample, answer);
                }
            }


            int transfer = 1;
            for (int i = n - 1; i >= 0; --i)
            {
                idx[i] += transfer;
                if (idx[i] == n)
                {
                    idx[i] = 0;
                }
                else
                {
                    break;
                }
            }
        }

        wrt_samples.Flush();

        return;
    }



    private static string Join(params string[] sx)
    {
        return string.Join(" ", sx).Replace(" ?", "?");
    }

    private static bool IsGoodObject(string obj_str)
    {
        foreach (string stopword in "кажд весь вся всю все день год месяц минуту секунду час ночь сутки год время шагу деру".Split())
        {
            if (obj_str.Contains(stopword))
            {
                return false;
            }
        }

        return true;
    }


    private static bool IsTimeNoun(string o)
    {
        return "ночью утром днем вечером зимой весной летом осенью".Split().Contains(o.ToLower());
    }


    private static List<SolarixGrammarEngineNET.SyntaxTreeNode> GetTerms(SolarixGrammarEngineNET.SyntaxTreeNode n)
    {
        List<SolarixGrammarEngineNET.SyntaxTreeNode> res = new List<SolarixGrammarEngineNET.SyntaxTreeNode>();
        res.Add(n);

        foreach (var child in n.leafs)
        {
            res.AddRange(GetTerms(child));
        }

        return res;
    }

    private static string TermToString(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode term)
    {
        int id_entry = term.GetEntryID();

        if (gren.GetEntryName(id_entry) == "???")
        {
            return term.GetWord();
        }

        string res_word = gren.RestoreCasing(id_entry, term.GetWord());

        return res_word;
    }


    private static string TermsToString(SolarixGrammarEngineNET.GrammarEngine2 gren, IEnumerable<SolarixGrammarEngineNET.SyntaxTreeNode> terms)
    {
        return string.Join(" ", terms.Select(z => TermToString(gren, z)));
    }

    private static string TermsToString(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode term)
    {
        return TermToString(gren, term);
    }


    static HashSet<string> where_to_adverbs;
    static bool IsWhereToAdverb(string adverb)
    {
        if (where_to_adverbs == null)
        {
            where_to_adverbs = new HashSet<string>();
            foreach (string a in " туда вправо влево вверх вниз вперед назад левее левей правее правей ниже пониже повыше выше".Split())
            {
                where_to_adverbs.Add(a);
            }
        }

        return where_to_adverbs.Contains(adverb);
    }


    static HashSet<string> be_verbs;
    static bool IsBeVerb(string v)
    {
        if (be_verbs == null)
        {
            be_verbs = new HashSet<string>();
            foreach (string w in "быть бывать стать становиться оказаться оказываться казаться показаться выглядеть".Split())
            {
                be_verbs.Add(w);
            }
        }

        return be_verbs.Contains(v);
    }



    static bool IsPunkt(string word)
    {
        return word.Length > 0 && char.IsPunctuation(word[0]);
    }

    static string GetNodeLemma(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode root)
    {
        int entry_id = root.GetEntryID();
        string lemma = gren.GetEntryName(entry_id);
        if (string.IsNullOrEmpty(lemma))
        {
            return root.GetWord();
        }
        else
        {
            return lemma;
        }
    }


    static string RebuildVerb2(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode v_node, string qword)
    {
        if (string.IsNullOrEmpty(qword))
        {
            return null;
        }

        List<int> coords = new List<int>();
        List<int> states = new List<int>();

        coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru);
        states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru);

        if (v_node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.TENSE_ru) == SolarixGrammarEngineNET.GrammarEngineAPI.PAST_ru)
        {
            // Для глагола в прошедшем времени надо указать род подлежащего - средний или мужской
            coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.GENDER_ru);
            if (qword.StartsWith("ч"))
            {
                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NEUTRAL_GENDER_ru);
            }
            else
            {
                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.MASCULINE_GENDER_ru);
            }
        }


        // Смена лица для глагола не-прошедшего времени:
        // Я продаю.
        // Кто продает?
        if (v_node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.TENSE_ru) != SolarixGrammarEngineNET.GrammarEngineAPI.PAST_ru)
        {
            coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru);
            states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_3_ru);
        }


        foreach (var p in v_node.GetPairs())
        {
            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.TENSE_ru ||
                p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.VERB_FORM_ru)
            {
                coords.Add(p.CoordID);
                states.Add(p.StateID);
            }
        }

        string v2 = "";
        List<string> fx = SolarixGrammarEngineNET.GrammarEngine.sol_GenerateWordformsFX(gren.GetEngineHandle(), v_node.GetEntryID(), coords, states);
        if (fx != null && fx.Count > 0)
        {
            v2 = fx[0].ToLower();
        }
        else
        {
            v2 = null;
        }

        return v2;
    }


    static string GetWhichQword4Obj(FootPrintToken ft)
    {
        string qword = "";

        if (ft.Match("inanim,fam,sing,acc"))
        {
            qword = "какую";
        }
        else if (ft.Match("inanim,masc,sing,acc"))
        {
            qword = "какой";
        }
        else if (ft.Match("inanim,neut,sing,acc"))
        {
            qword = "какое";
        }
        else if (ft.Match("inanim,pl,acc"))
        {
            qword = "какие";
        }
        else if (ft.Match("anim,masc,sing,acc"))
        {
            qword = "какого";
        }
        else if (ft.Match("anim,fam,sing,acc"))
        {
            qword = "какую";
        }
        else if (ft.Match("anim,neut,sing,acc"))
        {
            qword = "какое";
        }
        else if (ft.Match("anim,pl,acc"))
        {
            qword = "каких";
        }


        if (ft.Match("fam,sing,gen"))
        {
            qword = "какой";
        }
        else if (ft.Match("masc,sing,gen"))
        {
            qword = "какого";
        }
        else if (ft.Match("neut,sing,gen"))
        {
            qword = "какого";
        }
        else if (ft.Match("pl,gen"))
        {
            qword = "каких";
        }

        return qword;
    }


    static string GetWhichQword4Subject(FootPrintToken ft)
    {
        if (stop_adjs.Contains(ft.GetWord()))
            return null;

        string qword = "";
        if (ft.Match("fem,sing"))
        {
            qword = "какая";
        }
        else if (ft.Match("masc,sing"))
        {
            qword = "какой";
        }
        else if (ft.Match("neut,sing"))
        {
            qword = "какое";
        }
        else if (ft.Match("pl"))
        {
            qword = "какие";
        }

        return qword;
    }


    static string GetWhichQword4Instr(FootPrintToken ft)
    {
        string qword = "";
        if (ft.Match("fam,sing"))
        {
            qword = "какой";
        }
        else if (ft.Match("masc,sing"))
        {
            qword = "каким";
        }
        else if (ft.Match("neut,sing"))
        {
            qword = "каким";
        }
        else if (ft.Match("pl"))
        {
            qword = "какими";
        }

        return qword;
    }


    static string GetQuestionWordForAdverb(string a0)
    {
        return AdverbCategory.GetQuestionWordForAdverb(a0);
    }


    static string GetSubjectQuestion(FootPrintToken ft)
    {
        if (ft.Match("anim"))
        {
            return "кто";
        }
        else if (ft.Match("inanim"))
        {
            return "что";
        }
        else if (ft.Match("pr,1"))
        {
            return "кто";
        }
        else if (ft.Match("pr,2"))
        {
            return "кто";
        }

        return null;
    }


    static string GetAccusObjectQuestion(FootPrintToken ft)
    {
        // Вопросы к прямому дополнению.
        string qword = null; ;
        if (ft.Match("inanim")) // ребят
        {
            qword = "что";
        }
        else if (ft.Match("anim"))
        {
            qword = "кого";
        }

        return qword;
    }

    static string GetDativeObjectQuestion(FootPrintToken ft)
    {
        // Вопросы к прямому дополнению.
        string qword = null; ;
        if (ft.Match("inanim"))
        {
            qword = "чему";
        }
        else if (ft.Match("anim"))
        {
            qword = "кому";
        }

        return qword;
    }

    static string GetInstrObjectQuestion(FootPrintToken ft)
    {
        string qword = null;
        if (ft.Match("anim"))
        {
            qword = "кем";
        }
        else
        {
            qword = "чем";
        }
        return qword;
    }

    static string Preprocess(string phrase, SolarixGrammarEngineNET.GrammarEngine2 gren)
    {
        return preprocessor.Preprocess(phrase, gren);
    }

    static bool do_change_person = false;


    static string ChangePronounTo(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode node, string to_person)
    {
        if (do_change_person)
        {
            List<int> coords = new List<int>();
            List<int> states = new List<int>();

            if (to_person == "1s")
            {
                coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru);
                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru);

                coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru);
                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_1_ru);
            }
            else if (to_person == "2s")
            {
                coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru);
                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru);

                coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru);
                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_2_ru);
            }
            else if (to_person == "3s")
            {
                coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru);
                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru);

                coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru);
                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_2_ru);
            }
            else
            {
                throw new ArgumentException("to_person");
            }


            coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.CASE_ru);
            states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NOMINATIVE_CASE_ru);

            string new_word = "";
            List<string> fx = SolarixGrammarEngineNET.GrammarEngine.sol_GenerateWordformsFX(gren.GetEngineHandle(), node.GetEntryID(), coords, states);
            if (fx != null && fx.Count > 0)
            {
                new_word = fx[0].ToLower();
            }
            else
            {
                new_word = null;
            }

            return new_word;
        }
        else
        {
            return node.GetWord();
        }
    }



    static string ChangeVerbTo(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode node, string to_person)
    {
        if (do_change_person)
        {

            List<int> coords = new List<int>();
            List<int> states = new List<int>();

            coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.TENSE_ru);
            states.Add(node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.TENSE_ru));

            if (node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.TENSE_ru) != SolarixGrammarEngineNET.GrammarEngineAPI.PAST_ru)
            {
                if (to_person == "1s")
                {
                    coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru);
                    states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru);

                    coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru);
                    states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_1_ru);
                }
                else if (to_person == "2s")
                {
                    coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru);
                    states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru);

                    coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru);
                    states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_2_ru);
                }
                else if (to_person == "3s")
                {
                    coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru);
                    states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru);

                    coords.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru);
                    states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_2_ru);
                }
                else
                {
                    throw new ArgumentException("to_person");
                }
            }


            foreach (var p in node.GetPairs())
            {
                if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.TENSE_ru ||
                    p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.VERB_FORM_ru)
                {
                    coords.Add(p.CoordID);
                    states.Add(p.StateID);
                }

                if (to_person == "1s" || to_person == "2s")
                {
                    if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.GENDER_ru)
                    {
                        coords.Add(p.CoordID);
                        states.Add(p.StateID);
                    }
                }

            }

            string v2 = "";
            List<string> fx = SolarixGrammarEngineNET.GrammarEngine.sol_GenerateWordformsFX(gren.GetEngineHandle(), node.GetEntryID(), coords, states);
            if (fx != null && fx.Count > 0)
            {
                v2 = fx[0].ToLower();
            }
            else
            {
                v2 = null;
            }

            return v2;
        }
        else
        {
            return node.GetWord();
        }
    }


    static string ChangePersonTo(string sbj)
    {
        if (do_change_person)
        {
            if (sbj == "я")
            {
                return "2s";
            }
            else if (sbj == "ты")
            {
                return "1s";
            }
            else
            {
                return null;
            }
        }
        else
        {
            return sbj;
        }
    }


    static int nb_processed = 0;
    static HashSet<string> processed_phrases = new HashSet<string>();
    static void ProcessSentence(string phrase, SolarixGrammarEngineNET.GrammarEngine2 gren, int max_len)
    {
        nb_processed += 1;

        if (phrase.Length > 2 && phrase.Last() != '?')
        {
            bool used = false;

            if (phrase.Last() == '.' || phrase.Last() == '!')
            {
                phrase = phrase.Substring(0, phrase.Length - 1);
            }

            if (!processed_phrases.Contains(phrase))
            {
                processed_phrases.Add(phrase);

                string phrase2 = Preprocess(phrase, gren);

                int id_language = SolarixGrammarEngineNET.GrammarEngineAPI.RUSSIAN_LANGUAGE;
                SolarixGrammarEngineNET.GrammarEngine.MorphologyFlags morph_flags = SolarixGrammarEngineNET.GrammarEngine.MorphologyFlags.SOL_GREN_COMPLETE_ONLY | SolarixGrammarEngineNET.GrammarEngine.MorphologyFlags.SOL_GREN_MODEL;
                SolarixGrammarEngineNET.GrammarEngine.SyntaxFlags syntax_flags = SolarixGrammarEngineNET.GrammarEngine.SyntaxFlags.DEFAULT;
                int MaxAlt = 40;
                int constraints = 600000 | (MaxAlt << 22);

                using (SolarixGrammarEngineNET.AnalysisResults linkages = gren.AnalyzeSyntax(phrase2, id_language, morph_flags, syntax_flags, constraints))
                {
                    if (linkages.Count == 3)
                    {
                        SolarixGrammarEngineNET.SyntaxTreeNode root = linkages[1];
                        List<SolarixGrammarEngineNET.SyntaxTreeNode> terms = GetTerms(root).OrderBy(z => z.GetWordPosition()).ToList();

                        FootPrint footprint = new FootPrint(gren, terms);

                        string predicate_lemma = GetNodeLemma(gren, root);
                        //if (!IsBeVerb(predicate_lemma))
                        {
                            #region Его зовут Лешка.
                            if (footprint.Match("acc,sing зовут n,nom,sing"))
                            {
                                used = true;

                                string whom = TermsToString(gren, terms[0]);
                                string answer = TermsToString(gren, terms[2]);

                                if (do_change_person)
                                {
                                    if (StringExtender.EqCI(whom, "меня"))
                                    {
                                        whom = "тебя";
                                    }
                                    else if (StringExtender.EqCI(whom, "тебя"))
                                    {
                                        whom = "меня";
                                    }
                                }


                                WritePermutationsQA2(phrase, answer, "как", whom, "зовут");
                            }
                            #endregion Его зовут Лешка.


                            #region Я не ношу подделки!
                            if (footprint.Match("pr,1|2,nom,sing neg v,vf1,acc,~gen n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms.Skip(3).Take(1)); // подделки
                                var v_node = terms[2]; // ношу

                                string qword = null;
                                string answer = null;

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[2], to_person);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;

                                        // Я не ношу подделки
                                        // Ты носишь подделки?
                                        // Нет.
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s2, v2, o);
                                        //WritePermutationsQA2(phrase, answer, s2, v2+" ли", o);

                                        // Вопрос к дополнению:
                                        // Что ты не носишь?
                                        qword = GetAccusObjectQuestion(footprint[3]);
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, s2, v3, qword);
                                        }
                                    }

                                    // Вопрос к подлежащему:
                                    // Кто не носит подделки?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v3, o);
                                    }
                                }
                            }
                            #endregion Я не ношу подделки!


                            #region Габариты я не знаю.
                            if (footprint.Match("n,acc pr,1|2,nom,sing neg v,vf1,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[1]); // я
                                string o = TermsToString(gren, terms.Take(1)); // габариты
                                var v_node = terms[3]; // знаю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[1], to_person);
                                    string v2 = ChangeVerbTo(gren, v_node, to_person);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        string qword = null;
                                        string answer = null;

                                        // Габариты я не знаю
                                        // Ты знаешь габариты?
                                        // Нет.
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s2, v2, o);
                                        //WritePermutationsQA2(phrase, answer, s2, v2+" ли", o);

                                        // Вопрос к дополнению:
                                        // Что ты не знаешь?
                                        qword = GetAccusObjectQuestion(footprint[0]);
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, s2, v3, qword);
                                        }

                                        // Вопрос к подлежащему:
                                        // Кто не знает габариты?
                                        qword = GetSubjectQuestion(footprint[1]);
                                        v2 = RebuildVerb2(gren, v_node, qword);
                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            v3 = "не " + v2;
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v3, o);
                                        }
                                    }
                                }
                            }
                            #endregion Габариты я не знаю.


                            #region Я габариты не знаю.
                            if (footprint.Match("pr,1|2,nom,sing n,acc neg v,vf1,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms[1]); // габариты
                                var v_node = terms[3]; // знаю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, v_node, to_person);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        string qword = null;
                                        string answer = null;

                                        // Я габариты не знаю
                                        // Ты знаешь габариты?
                                        // Нет.
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s2, v2, o);
                                        //WritePermutationsQA2(phrase, answer, s2, v2+" ли", o);

                                        // Вопрос к дополнению:
                                        // Что ты не знаешь?
                                        qword = GetAccusObjectQuestion(footprint[1]);
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, s2, v3, qword);
                                        }

                                        // Вопрос к подлежащему:
                                        // Кто не знает габариты?
                                        qword = GetSubjectQuestion(footprint[0]);
                                        v2 = RebuildVerb2(gren, v_node, qword);
                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            v3 = "не " + v2;
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v3, o);
                                        }
                                    }
                                }
                            }
                            #endregion Габариты я не знаю.


                            #region Габариты не знаю я.
                            if (footprint.Match("n,acc neg v,vf1,acc pr,1|2,nom,sing"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[3]); // я
                                string o = TermsToString(gren, terms[0]); // габариты
                                var v_node = terms[2]; // знаю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[3], to_person);
                                    string v2 = ChangeVerbTo(gren, v_node, to_person);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        string qword = null;
                                        string answer = null;

                                        // Габариты не знаю я.
                                        // Ты знаешь габариты?
                                        // Нет.
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s2, v2, o);
                                        //WritePermutationsQA2(phrase, answer, s2, v2+" ли", o);

                                        // Вопрос к дополнению:
                                        // Что ты не знаешь?
                                        qword = GetAccusObjectQuestion(footprint[0]);
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, s2, v3, qword);
                                        }

                                        // Вопрос к подлежащему:
                                        // Кто не знает габариты?
                                        qword = GetSubjectQuestion(footprint[3]);
                                        v2 = RebuildVerb2(gren, v_node, qword);
                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            v3 = "не " + v2;
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v3, o);
                                        }
                                    }
                                }
                            }
                            #endregion Габариты я не знаю.


                            #region Почтой я не отправляю.
                            if (footprint.Match("n,instr pr,1|2,nom,sing neg v,vf1"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[1]); // я
                                string o = TermsToString(gren, terms.Take(1)); // почтой
                                var v_node = terms[3]; // отправляю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[1], to_person);
                                    string v2 = ChangeVerbTo(gren, v_node, to_person);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        string qword = null;
                                        string answer = null;

                                        // Почтой я не отправляю
                                        // Ты отправляешь почтой?
                                        // Нет.
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s2, v2, o);
                                        //WritePermutationsQA2(phrase, answer, s2, v2+" ли", o);


                                        // Вопрос к подлежащему:
                                        // Кто не знает габариты?
                                        qword = GetSubjectQuestion(footprint[1]);
                                        v2 = RebuildVerb2(gren, v_node, qword);
                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            v3 = "не " + v2;
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v3, o);
                                        }
                                    }
                                }
                            }
                            #endregion Почтой я НЕ отправляю.


                            #region Я почтой не отправляю.
                            if (footprint.Match("pr,1|2,nom,sing n,instr neg v,vf1"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms[1]); // почтой
                                var v_node = terms[3]; // отправляю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, v_node, to_person);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        string qword = null;
                                        string answer = null;

                                        // Я почтой не отправляю
                                        // Ты отправляешь почтой?
                                        // Нет.
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s2, v2, o);
                                        //WritePermutationsQA2(phrase, answer, s2, v2+" ли", o);


                                        // Вопрос к подлежащему:
                                        // Кто не отправляет почтой?
                                        qword = GetSubjectQuestion(footprint[0]);
                                        v2 = RebuildVerb2(gren, v_node, qword);
                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            v3 = "не " + v2;
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v3, o);
                                        }
                                    }
                                }
                            }
                            #endregion Почтой я НЕ отправляю.


                            #region я не отправляю почтой.
                            if (footprint.Match("pr,1|2,nom,sing neg v,vf1 n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms[3]); // почтой
                                var v_node = terms[2]; // отправляю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, v_node, to_person);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        string qword = null;
                                        string answer = null;

                                        // Я почтой не отправляю
                                        // Ты отправляешь почтой?
                                        // Нет.
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s2, v2, o);
                                        //WritePermutationsQA2(phrase, answer, s2, v2+" ли", o);


                                        // Вопрос к подлежащему:
                                        // Кто не отправляет почтой?
                                        qword = GetSubjectQuestion(footprint[0]);
                                        v2 = RebuildVerb2(gren, v_node, qword);
                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            v3 = "не " + v2;
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v3, o);
                                        }
                                    }
                                }
                            }
                            #endregion я не отправляю почтой.


                            #region Не отправляю я почтой.
                            if (footprint.Match("neg v,vf1 pr,1|2,nom,sing n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[2]); // я
                                string o = TermsToString(gren, terms[3]); // почтой
                                var v_node = terms[1]; // отправляю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[2], to_person);
                                    string v2 = ChangeVerbTo(gren, v_node, to_person);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        string qword = null;
                                        string answer = null;

                                        // Не отправляю я почтой.
                                        // Ты отправляешь почтой?
                                        // Нет.
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s2, v2, o);
                                        //WritePermutationsQA2(phrase, answer, s2, v2+" ли", o);


                                        // Вопрос к подлежащему:
                                        // Кто не отправляет почтой?
                                        qword = GetSubjectQuestion(footprint[2]);
                                        v2 = RebuildVerb2(gren, v_node, qword);
                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            v3 = "не " + v2;
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v3, o);
                                        }
                                    }
                                }
                            }
                            #endregion Не отправляю я почтой.


                            #region Я гарантирую возврат денег
                            if (footprint.Match("pr,1|2,nom,sing v,vf1,acc,~gen n,acc n,gen"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms.Skip(2).Take(2)); // возврат денег
                                var v_node = terms[1]; // гарантирую

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[1], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я гарантирую возврат денег
                                    // Ты гарантируешь возврат денег?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2, o);

                                    // Вопрос к дополнению:
                                    // Что ты гарантируешь?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s2, v2, qword);
                                    }

                                    // Вопрос к подлежащему:
                                    // Кто гарантирует возврат денег?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Я гарантирую возврат денег


                            #region Я жду Ваших предложений
                            if (footprint.Match("pr,1|2,nom,sing v,vf1,gen adj,gen n,gen"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms.Skip(2).Take(2)); // Ваших предложений
                                var v_node = terms[1]; // гарантирую

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[1], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я жду Ваших предложений
                                    // Ты ждешь ваших предложений?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2, o);

                                    // Вопрос к дополнению:
                                    // Что ты ждешь?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s2, v2, qword);
                                    }

                                    // Вопросы "какой? etc" к прямому дополнению
                                    // Каких предложений ты ждешь?
                                    qword = GetWhichQword4Obj(footprint[3]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        string o2 = qword + " " + TermsToString(gren, terms[3]); // каких предложений?
                                        answer = TermsToString(gren, terms[2]); // ваших
                                        WritePermutationsQA2(phrase, answer, o2, v2, s2);
                                    }


                                    // Вопрос к подлежащему:
                                    // Кто ждет ваших предложений?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Я жду Ваших предложений


                            #region Я помню чудное мгновенье.
                            if (footprint.Match("pr,1|2,nom,sing v,vf1,acc adj,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms.Skip(2).Take(2)); // чудное мгновенье
                                var v_node = terms[1]; // помню

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[1], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я помню чудное мгновенье.
                                    // Ты помнишь чудное мгновенье?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2, o);

                                    // Вопрос к дополнению:
                                    // Что ты помнишь?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s2, v2, qword);
                                    }

                                    // Вопросы "какой? etc" к прямому дополнению
                                    // Какое мгновенье ты помнишь?
                                    qword = GetWhichQword4Obj(footprint[3]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        string o2 = qword + " " + TermsToString(gren, terms[3]); // какое мгновенье
                                        answer = TermsToString(gren, terms[2]); // чудное
                                        WritePermutationsQA2(phrase, answer, o2, v2, s2);
                                    }

                                    // Вопрос к подлежащему:
                                    // Кто помнит чудное мгновенье?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion


                            #region Я нахожусь в Краснодаре.
                            if (footprint.Match("pr,1|2,nom,sing v,vf1 prep n"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                var v_node = terms[1]; // нахожусь
                                string pn = TermsToString(gren, terms.Skip(2).Take(2)); // в Краснодаре

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[1], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я нахожусь в Краснодаре.
                                    // Ты находишься в Краснодаре?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2, pn);

                                    // Вопрос к подлежащему:
                                    // Кто находится в Краснодаре?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, pn);
                                    }
                                }
                            }
                            #endregion Я нахожусь в Краснодаре.


                            #region И я читал.
                            if (footprint.Match("pr,1|2,nom,sing v,vf1"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                var v_node = terms[1]; // читал

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[1], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я читаю.
                                    // Ты читаешь?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2);

                                    // Вопрос к подлежащему:
                                    // Кто читает?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2);
                                    }
                                }
                            }
                            #endregion И я читал.


                            #region Я только продаю!
                            if (footprint.Match("pr,1|2,nom,sing adv v,vf1"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string a = TermsToString(gren, terms[1]); // только
                                var v_node = terms[2]; // продаю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[2], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я только продаю.
                                    // Ты только продаешь?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, a, v2);

                                    // Вопрос к подлежащему:
                                    // Кто продает?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, a, v2);
                                    }
                                }
                            }
                            #endregion Я только продаю!


                            #region Радовался я недолго.
                            if (footprint.Match("v,vf1 pr,1|2,nom,sing adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[1]); // я
                                string a = TermsToString(gren, terms[2]); // недолго
                                var v_node = terms[0]; // радовался

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[1], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[0], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Радовался я недолго.
                                    // Ты недолго радовался?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, a, v2);

                                    // Вопрос к подлежащему:
                                    // Кто недолго радовался?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, a, v2);
                                    }
                                }
                            }
                            #endregion Радовался я недолго.


                            #region Теперь я знаю.
                            if (footprint.Match("adv pr,1|2,nom,sing v,vf1"))
                            {
                                used = true;

                                string a = TermsToString(gren, terms[0]); // теперь
                                string s = TermsToString(gren, terms[1]); // я
                                var v_node = terms[2]; // знаю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[1], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[2], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Теперь я знаю.
                                    // Теперь ты знаешь?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, a, v2);

                                    // Вопрос к подлежащему:
                                    // Кто теперь знает?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, a, v2);
                                    }
                                }
                            }
                            #endregion Теперь я знаю.


                            #region Я не курю.
                            if (footprint.Match("pr,1|2,nom,sing neg v,vf1"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                var v_node = terms[2]; // курю

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[2], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я не курю.
                                    // Ты куришь?
                                    // Нет.
                                    answer = "нет";
                                    WritePermutationsQA2(phrase, answer, s2, v2);
                                    //WritePermutationsQA2(phrase, answer, s2, v2+" ли");

                                    // Вопрос к подлежащему:
                                    // Кто не курит?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        string v3 = "не " + v2;
                                        answer = s2;
                                        WritePermutationsQA2(phrase, answer, qword, v3);
                                    }
                                }
                            }
                            #endregion Я не курю.


                            #region Я заклеил моментом.
                            if (footprint.Match("pr,1|2,nom,sing v,vf1,instr n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms[2]); // моментом
                                var v_node = terms[1]; // заклеил

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[1], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я заклеил моментом.
                                    // Ты заклеил моментом?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2, o);

                                    // Вопрос к дополнению:
                                    qword = GetInstrObjectQuestion(footprint[2]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s2, v2, qword);
                                    }

                                    // Вопрос к подлежащему:
                                    // Кто заклеил моментом?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Я заклеил моментом.


                            #region Я ищу работу.
                            if (footprint.Match("pr,1|2,nom,sing v,vf1,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms[2]); // работу
                                var v_node = terms[1]; // ищу

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[1], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я ищу работу.
                                    // Ты ищешь работу?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2, o);

                                    // Вопрос к дополнению:
                                    // Что ты ищешь?
                                    qword = GetAccusObjectQuestion(footprint[2]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s2, v2, qword);
                                    }

                                    // Вопрос к подлежащему:
                                    // Кто ищет работу?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Я ищу работу.


                            #region Коробку я потерял...
                            if (footprint.Match("n,acc pr,1|2,nom,sing v,vf1,acc"))
                            {
                                used = true;

                                string o = TermsToString(gren, terms[0]); // коробку
                                string s = TermsToString(gren, terms[1]); // я
                                var v_node = terms[2]; // потерял

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[1], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[2], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Коробку я потерял.
                                    // Ты потерял коробку?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2, o);

                                    // Вопрос к дополнению:
                                    // Что ты потерял?
                                    qword = GetAccusObjectQuestion(footprint[0]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s2, v2, qword);
                                    }

                                    // Вопрос к подлежащему:
                                    // Кто потерял коробку?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Коробку я потерял...


                            #region Я жаловался коллегам.
                            if (footprint.Match("pr,1|2,nom,sing v,vf1,dat n,dat"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // я
                                string o = TermsToString(gren, terms[2]); // коллегам
                                var v_node = terms[1]; // жаловался

                                string to_person = ChangePersonTo(s);
                                if (!string.IsNullOrEmpty(to_person))
                                {
                                    string s2 = ChangePronounTo(gren, terms[0], to_person);
                                    string v2 = ChangeVerbTo(gren, terms[1], to_person);

                                    string qword = null;
                                    string answer = null;

                                    // Я жаловался коллегам.
                                    // Ты жаловался коллегам?
                                    // Да.
                                    answer = "да";
                                    WritePermutationsQA2(phrase, answer, s2, v2, o);

                                    // Вопрос к дополнению:
                                    // Кому ты жаловался?
                                    qword = GetDativeObjectQuestion(footprint[2]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s2, v2, qword);
                                    }

                                    // Вопрос к подлежащему:
                                    // Кто жаловался коллегам?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Я жаловался коллегам.


                            #region Петя ничуть не смутился.
                            if (footprint.Match("n,nom adv neg v,vf1"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // Петя
                                var v_node = terms[3]; // смутился
                                string v = TermsToString(gren, terms[3]); // смутился

                                string answer = "нет";
                                WritePermutationsQA2(phrase, answer, s, v);
                                //WritePermutationsQA2(phrase, answer, s, v+" ли");


                                string qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, "не " + v2);
                                }

                            }
                            #endregion Петя ничуть не смутился.


                            #region Юбка не мнется совершенно!
                            if (footprint.Match("n,nom neg v,vf1 adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // Юбка
                                var v_node = terms[2]; // мнется
                                string v = TermsToString(gren, terms[2]); // мнется

                                string answer = "нет";
                                WritePermutationsQA2(phrase, answer, s, v);
                                //WritePermutationsQA2(phrase, answer, s, v+" ли");

                                string qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, "не " + v2);
                                }
                            }
                            #endregion Юбка не мнется совершенно!


                            #region Китайцы не были моряками.
                            if (footprint.Match("n,nom neg v,vf1 n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[0]); // Китайцы
                                var v_node = terms[2]; // были
                                string v = TermsToString(gren, terms[2]); // были
                                string o = TermsToString(gren, terms[3]); // моряками

                                string answer = "нет";
                                WritePermutationsQA2(phrase, answer, s, v, o);
                                //WritePermutationsQA2(phrase, answer, s, v+" ли", o);

                                string qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, "не " + v2, o);
                                }
                            }
                            #endregion Китайцы не были моряками.


                            #region Не мнется юбка совершенно!
                            if (footprint.Match("neg v,vf1 n,nom adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[2]); // Юбка
                                var v_node = terms[1]; // мнется
                                string v = TermsToString(gren, terms[1]); // мнется

                                string answer = "нет";
                                WritePermutationsQA2(phrase, answer, s, v);
                                //WritePermutationsQA2(phrase, answer, s, v+" ли");

                                string qword = GetSubjectQuestion(footprint[2]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, "не " + v2);
                                }

                            }
                            #endregion Юбка не мнется совершенно!


                            #region Ранее автомобиль не эксплуатировался.
                            if (footprint.Match("adv n,nom neg v,vf1"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms[1]); // автомобиль
                                var v_node = terms[3]; // эксплуатировался
                                string v = TermsToString(gren, terms[3]); // мнется

                                string answer = "нет";
                                WritePermutationsQA2(phrase, answer, s, v);
                                //WritePermutationsQA2(phrase, answer, s, v+" ли");

                                string qword = GetSubjectQuestion(footprint[1]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, "не " + v2);
                                }

                            }
                            #endregion Ранее автомобиль не эксплуатировался.


                            #region Электронное зажигание упрощает запуск инструмента
                            if (footprint.Match("adj,nom n,nom v,acc,~gen n,acc n,gen"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // Электронное зажигание
                                string v = TermsToString(gren, terms[2]); // упрощает
                                string o = TermsToString(gren, terms.Skip(3).Take(2)); // запуск инструмента
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Что упрощает электронное зажигание?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что упрощает запуск инструмента?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }


                                    // Вопрос к атрибуту подлежащего:
                                    // Какое зажигание упрощает запуск инструмента?
                                    // ^^^^^
                                    qword = GetWhichQword4Subject(footprint[1]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        string s2 = qword + " " + TermsToString(gren, terms[1]); // какое зажигание
                                        answer = TermsToString(gren, terms[0]); // электронное

                                        WritePermutationsQA2(phrase, answer, s2, v, o);
                                    }

                                }
                            }
                            #endregion Электронное зажигание упрощает запуск инструмента


                            #region Процессор платы имеет пятифазное питание
                            if (footprint.Match("n,nom n,gen v,acc,~gen adj,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // Процессор платы
                                string v = TermsToString(gren, terms[2]); // имеет
                                string o = TermsToString(gren, terms.Skip(3).Take(2)); // пятифазное питание
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Что имеет процессор платы?
                                    qword = GetAccusObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что имеет пятифазное питание?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }



                                    // Вопросы "какой? etc" к прямому дополнению
                                    // Какое питание имеет процессор платы?
                                    qword = GetWhichQword4Obj(footprint[4]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        string o2 = qword + " " + TermsToString(gren, terms[4]); // какое питание
                                        answer = TermsToString(gren, terms[3]); // пятифазное
                                        WritePermutationsQA2(phrase, answer, o2, v, s);
                                    }

                                }
                            }
                            #endregion Процессор платы имеет пятифазное питание


                            #region Ностальгирующая публика принимала певца радушно.
                            if (footprint.Match("adj,nom n,nom v,acc n,acc adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // Ностальгирующая публика
                                string v = TermsToString(gren, terms[2]); // принимала
                                string o = TermsToString(gren, terms.Skip(3).Take(1)); // певца
                                string a = TermsToString(gren, terms[4]); // радушно
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Кого принимала радушно ностальгирующая публика?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, a, qword);

                                    // Вопросы к подлежащему
                                    // Кто принимал певца равнодушно?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a, o);
                                    }


                                    // Вопрос к наречному обстоятельству:
                                    // Как принимала певца ностальгирующая публика?
                                    // ^^^
                                    qword = GetQuestionWordForAdverb(a);

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = a;
                                        WritePermutationsQA2(phrase, answer, qword, v, s, o);
                                    }

                                }
                            }
                            #endregion Ностальгирующая публика принимала певца радушно.


                            #region Загадочная славянская душа выворачивается наизнанку.
                            if (footprint.Match("adj,nom adj,nom n,nom v adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(3)); // Загадочная славянская душа
                                string v = TermsToString(gren, terms[3]); // выворачивается
                                string a = TermsToString(gren, terms[4]); // наизнанку
                                var v_node = terms[3];

                                string answer = null;
                                string qword = null;

                                // Вопрос к подлежащему:
                                // Что выворачивается наизнанку?
                                // ^^^
                                qword = GetSubjectQuestion(footprint[2]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, a, v2);
                                }

                                // Вопрос к наречному обстоятельству:
                                // Как выворачивается загадочная славянская душа?
                                // ^^^
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, qword, v, s);
                                }
                            }
                            #endregion Загадочная славянская душа выворачивается наизнанку.


                            #region Вашему вниманию представляются Красивые номера
                            if (footprint.Match("adj,dat n,dat v,dat adj,nom n,nom"))
                            {
                                used = true;

                                string o = TermsToString(gren, terms.Take(2)); // Вашему вниманию
                                string v = TermsToString(gren, terms[2]); // представляются
                                string s = TermsToString(gren, terms.Skip(3).Take(2)); // Красивые номера
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Чему представляются красивые номера?
                                    qword = GetDativeObjectQuestion(footprint[1]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что представляется вашему вниманию?
                                    qword = GetSubjectQuestion(footprint[4]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Вашему вниманию представляются Красивые номера


                            #region Рассказывает сказки мама или папа
                            if (footprint.Match("v,acc n,acc n,nom conj n,nom"))
                            {
                                used = true;

                                string v = TermsToString(gren, terms[0]); // рассказывает
                                string o = TermsToString(gren, terms.Skip(1).Take(1)); // сказки
                                string s = TermsToString(gren, terms.Skip(2).Take(3)); // мама или папа
                                var v_node = terms[0];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Что рассказывает мама или папа?
                                    qword = GetAccusObjectQuestion(footprint[1]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Кто рассказывает сказки?
                                    qword = GetSubjectQuestion(footprint[4]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Рассказывает сказки мама или папа


                            #region Зарастают водорослями пруды и каналы.
                            if (footprint.Match("v,instr n,instr n,nom conj n,nom"))
                            {
                                used = true;

                                string v = TermsToString(gren, terms[0]); // зарастают
                                string o = TermsToString(gren, terms.Skip(1).Take(1)); // водорослями
                                string s = TermsToString(gren, terms.Skip(2).Take(3)); // пруды и каналы
                                var v_node = terms[0];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Чем зарастают пруды и каналы?
                                    qword = GetInstrObjectQuestion(footprint[0]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что зарастает водорослями?
                                    qword = GetSubjectQuestion(footprint[4]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Зарастают водорослями пруды и каналы.


                            #region Фотографии и цена соответствуют действительности
                            if (footprint.Match("n,nom conj n,nom v,dat n,dat"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(3)); // Фотографии и цена
                                string v = TermsToString(gren, terms[3]); // соответствуют
                                string o = TermsToString(gren, terms.Skip(4).Take(1)); // действительности
                                var v_node = terms[3];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Чему соответствуют фотографии и цена?
                                    qword = GetDativeObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что соответствует действительности?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Фотографии и цена соответствуют действительности


                            #region Мотор и коробка идеально работают
                            if (footprint.Match("n,nom conj n,nom adv v"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(3)); // Мотор и коробка
                                string a = TermsToString(gren, terms[3]); // идеально
                                string v = TermsToString(gren, terms[4]); // работают
                                var v_node = terms[4];

                                string qword = null;
                                string answer = null;

                                // Вопросы к подлежащему
                                // Что работает идеально?
                                qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, a);
                                }


                                // Вопрос к наречному обстоятельству:
                                // Как работают мотор и коробка?
                                // ^^^
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, qword, v, s);
                                }
                            }
                            #endregion Мотор и коробка идеально работают


                            #region Басаев угрожает церквям и детям.
                            if (footprint.Match("n,nom v,dat n,dat conj n,dat"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // Басаев
                                string v = TermsToString(gren, terms[1]); // угрожает
                                string o = TermsToString(gren, terms.Skip(2).Take(3)); // церквям и детям
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Чему угрожает Басаев?
                                    qword = GetDativeObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Кто угрожает церквям и детям?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Басаев угрожает церквям и детям.


                            #region Девушка снимет комнату или квартиру
                            if (footprint.Match("n,nom v,acc n,acc conj n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // Девушка
                                string v = TermsToString(gren, terms[1]); // снимет
                                string o = TermsToString(gren, terms.Skip(2).Take(3)); // комнату или квартиру
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Что снимет девушка?
                                    qword = GetAccusObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Кто снимет комнату или квартиру?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Девушка снимет комнату или квартиру


                            #region Возродившийся конкурс активно набирает обороты.
                            if (footprint.Match("adj,nom n,nom adv v,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // Возродившийся конкурс
                                string a = TermsToString(gren, terms[2]); // активно
                                string v = TermsToString(gren, terms[3]); // набирает
                                string o = TermsToString(gren, terms.Skip(4).Take(1)); // обороты
                                var v_node = terms[3];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Что активно набирает возродившийся конкурс?
                                    qword = GetAccusObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, a, v, qword);
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что активно набирает обороты?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, a, v2, o);
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }

                                    // Вопрос к наречному обстоятельству:
                                    // Как набирает обороты возродившийся конкурс?
                                    // ^^^
                                    qword = GetQuestionWordForAdverb(a);

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = a;
                                        WritePermutationsQA2(phrase, answer, qword, v, s, o);
                                    }

                                }
                            }
                            #endregion Возродившийся конкурс активно набирает обороты.


                            #region Внезапная острая боль перехватила дыхание.
                            if (footprint.Match("adj,nom adj,nom n,nom v,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(3)); // Внезапная острая боль
                                string v = TermsToString(gren, terms[3]); // перехватила
                                string o = TermsToString(gren, terms.Skip(4).Take(1)); // дыхание
                                var v_node = terms[3];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Что перехватила внезапная острая боль?
                                    qword = GetAccusObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что перехватило дыхание?
                                    qword = GetSubjectQuestion(footprint[2]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Внезапная острая боль перехватила дыхание.


                            #region Здесь вызревает янтарный игристый напиток.
                            if (footprint.Match("adv v adj,nom adj,nom n,nom"))
                            {
                                used = true;

                                string a = TermsToString(gren, terms[0]); // здесь
                                string v = TermsToString(gren, terms[1]); // вызревает
                                string s = TermsToString(gren, terms.Skip(2).Take(3)); // янтарный игристый напиток
                                var v_node = terms[1];

                                string answer = null;
                                string qword = null;

                                // Вопрос к подлежащему:
                                // Что здесь вызревает?
                                // ^^^
                                qword = GetSubjectQuestion(footprint[4]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, a, v2);
                                }

                                // Вопрос к наречному обстоятельству:
                                // Где вызревает янтарный игристый напиток?
                                // ^^^
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, qword, v, s);
                                }
                            }
                            #endregion Здесь вызревает янтарный игристый напиток.


                            #region Налоговый кодекс подвергается значительным правкам.
                            if (footprint.Match("adj,nom n,nom v,dat adj,dat n,dat"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // Налоговый кодекс
                                string v = TermsToString(gren, terms[2]); // подвергается
                                string o = TermsToString(gren, terms.Skip(3).Take(2)); // значительным правкам
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Налоговый кодекс подвергается чему?
                                    qword = GetDativeObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что подвергается правкам?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Налоговый кодекс подвергается значительным правкам.


                            #region Ласковый игривый малыш ждет своего хозяина
                            if (footprint.Match("adj,nom adj,nom n,nom v,acc adj,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(3)); // Ласковый игривый малыш
                                string v = TermsToString(gren, terms[3]); // ждет
                                string o = TermsToString(gren, terms.Skip(4).Take(2)); // своего хозяина
                                var v_node = terms[3];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Кого ждет ласковый игривый малыш?
                                    qword = GetAccusObjectQuestion(footprint[5]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Кто ждет своего хозяина?
                                    qword = GetSubjectQuestion(footprint[2]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Ласковый игривый малыш ждет своего хозяина


                            #region Зрителям смелость секретаря очень понравилась.
                            if (footprint.Match("n,dat n,nom n,gen adv v,dat"))
                            {
                                used = true;

                                string o = TermsToString(gren, terms.Take(1)); // зрителям
                                string s = TermsToString(gren, terms.Skip(1).Take(2)); // смелость секретаря
                                string a = TermsToString(gren, terms[3]); // очень
                                string v = TermsToString(gren, terms[4]); // понравилась
                                var v_node = terms[4];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Кому смелость секретаря очень понравилась?
                                    qword = GetDativeObjectQuestion(footprint[0]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, a, v, qword);

                                    // Вопросы к подлежащему
                                    // Что очень понравилось зрителям?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, a, v2, o);
                                    }


                                    // Вопрос к наречному обстоятельству
                                    // Как зрителям понравилась смелость секретаря?
                                    qword = GetQuestionWordForAdverb(a);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = a;
                                        WritePermutationsQA2(phrase, answer, qword, v, s);
                                    }
                                }
                            }
                            #endregion Зрителям смелость секретаря очень понравилась.


                            #region Выглядит такой термостакан очень достойно
                            if (footprint.Match("v adj,nom n,nom adv,a_modif adv"))
                            {
                                used = true;

                                string v = TermsToString(gren, terms[0]); // выглядит
                                string s = TermsToString(gren, terms.Skip(1).Take(2)); // такой термостакан
                                string a = TermsToString(gren, terms.Skip(3).Take(2)); // очень достойно
                                var v_node = terms[0];

                                string answer = null;
                                string qword = null;

                                // Вопрос к подлежащему:
                                // Что выглядит очень достойно?
                                // ^^^
                                qword = GetSubjectQuestion(footprint[2]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, a, v2);
                                }

                                // Вопрос к наречному обстоятельству:
                                // Как выглядит такой термостакан?
                                // ^^^
                                qword = GetQuestionWordForAdverb(TermsToString(gren, terms[4]));

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, qword, v, s);
                                }
                            }
                            #endregion Выглядит такой термостакан очень достойно


                            #region Яркие расцветки неизменно нравятся деткам
                            if (footprint.Match("adj,nom n,nom adv v,dat n,dat "))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // Яркие расцветки
                                string a = TermsToString(gren, terms[2]); // неизменно
                                string v = TermsToString(gren, terms[3]); // нравятся
                                string o = TermsToString(gren, terms.Skip(4).Take(1)); // деткам
                                var v_node = terms[3];

                                string answer = null;
                                string qword = null;

                                // Вопрос к подлежащему:
                                // что нравится деткам?
                                // ^^^
                                qword = GetSubjectQuestion(footprint[1]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, a, v2, o);
                                }

                                // Вопрос к наречному обстоятельству:
                                // Как нравятся деткам яркие расцветки?
                                // ^^^
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, qword, v, s, o);
                                }

                                // Вопросы к прямому дополнению:
                                // Кому нравятся яркие расцветки?
                                qword = GetDativeObjectQuestion(footprint[4]);
                                if (!string.IsNullOrEmpty(qword))
                                {
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);
                                    WritePermutationsQA2(phrase, answer, s, a, v, qword);
                                }

                            }
                            #endregion Яркие расцветки неизменно нравятся деткам


                            #region Трехмерное изображение создают регулируемые линзы
                            if (footprint.Match("adj,acc n,acc v,acc adj,nom n,nom"))
                            {
                                used = true;

                                string o = TermsToString(gren, terms.Take(2)); // трехмерное изображение
                                string v = TermsToString(gren, terms[2]); // создают
                                string s = TermsToString(gren, terms.Skip(3).Take(2)); // регулируемые линзы
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Что создают регулируемые линзы?
                                    qword = GetAccusObjectQuestion(footprint[1]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что создает трехмерное изображение?
                                    qword = GetSubjectQuestion(footprint[4]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Трехмерное изображение создают регулируемые линзы


                            #region Пленка имеет глубокий черный цвет
                            if (footprint.Match("n,nom v,acc adj,acc adj,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // пленка
                                string v = TermsToString(gren, terms[1]); // имеет
                                string o = TermsToString(gren, terms.Skip(2).Take(3)); // глубокий черный цвет
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Что имеет пленка?
                                    // ^^^
                                    qword = GetAccusObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что имеет глубокий черный цвет?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Пленка имеет глубокий черный цвет


                            #region Металлодетекторы обладают большой пропускной способностью
                            if (footprint.Match("n,nom v,instr adj,instr adj,instr n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // Металлодетекторы
                                string v = TermsToString(gren, terms[1]); // обладают
                                string o = TermsToString(gren, terms.Skip(2).Take(3)); // большой пропускной способностью
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Чем обладают металлодетекторы?
                                    // ^^^
                                    qword = GetInstrObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что обладает большой пропускной способностью?
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Металлодетекторы обладают большой пропускной способностью


                            #region Стальная рама отличается высокой прочностью
                            if (footprint.Match("adj,nom n,nom v,instr adj,instr n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // Стальная рама
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // отличается
                                string o = TermsToString(gren, terms.Skip(3).Take(2)); // высокой прочностью
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Чем отличается стальная рама?
                                    // ^^^
                                    qword = GetInstrObjectQuestion(footprint[4]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    // Что отличается высокой прочностью?
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Стальная рама отличается высокой прочностью


                            #region Продается очень мощный компьютер.
                            if (footprint.Match("v adv,a_modif adj,nom n,nom"))
                            {
                                used = true;

                                string v = TermsToString(gren, terms.Take(1)); // продается
                                string s = TermsToString(gren, terms.Skip(1).Take(3)); // очень мощный компьютер
                                var v_node = terms[0];

                                string answer = null;
                                string qword = null;

                                // Вопрос к атрибуту подлежащего:
                                // Какой компьютер продается?
                                // ^^^^^
                                qword = GetWhichQword4Subject(footprint[3]);
                                if (!string.IsNullOrEmpty(qword))
                                {
                                    string s2 = qword + " " + TermsToString(gren, terms.Skip(3).Take(1));
                                    answer = TermsToString(gren, terms.Skip(1).Take(2)); // очень мощный

                                    // Какой компьютер продается?
                                    WritePermutationsQA2(phrase, answer, s2, v);
                                }


                                // Вопрос к подлежащему:
                                // Что продается?
                                // ^^^
                                qword = GetSubjectQuestion(footprint[3]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2);
                                }
                            }

                            #endregion Продается очень мощный компьютер.


                            #region Очень мощный компьютер продается.
                            if (footprint.Match("adv,a_modif adj,nom n,nom v"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(3)); // очень мощный компьютер
                                string v = TermsToString(gren, terms.Skip(3).Take(1)); // продается
                                var v_node = terms[3];

                                string answer = null;
                                string qword = null;

                                // Вопрос к атрибуту подлежащего:
                                // Какой компьютер продается?
                                // ^^^^^
                                qword = GetWhichQword4Subject(footprint[2]);
                                if (!string.IsNullOrEmpty(qword))
                                {
                                    string s2 = qword + " " + TermsToString(gren, terms.Skip(2).Take(1));
                                    answer = TermsToString(gren, terms.Take(2)); // очень мощный

                                    // Какой компьютер продается?
                                    WritePermutationsQA2(phrase, answer, s2, v);
                                }


                                // Вопрос к подлежащему:
                                // Что продается?
                                // ^^^
                                qword = GetSubjectQuestion(footprint[2]);
                                string v2 = RebuildVerb2(gren, v_node, qword);
                                if (!string.IsNullOrEmpty(v2))
                                {
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2);
                                }
                            }
                            #endregion Продается очень мощный компьютер.


                            #region Модель выглядит очень гармонично.
                            if (footprint.Match("n,nom v adv,a_modif adv,adv_как"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // модель
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // выглядит
                                string a = TermsToString(gren, terms.Skip(2).Take(2)); // очень гармонично
                                var v_node = terms[1];

                                string answer = null;
                                string qword = null;

                                // Вопрос к обстоятельству
                                // Как выглядит модель?
                                qword = GetQuestionWordForAdverb(TermToString(gren, terms[3]));

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);


                                    // Вопрос к подлежащему:
                                    // Что выглядит очень гармонично?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Модель выглядит очень гармонично.


                            #region Брат пристально посмотрел на доктора.
                            if (footprint.Match("n,nom adv v prep n,acc,anim"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // брат
                                string a = TermsToString(gren, terms.Skip(1).Take(1)); // пристально
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // посмотрел
                                string pn = TermsToString(gren, terms.Skip(3).Take(2)); // на доктора
                                var v_node = terms[2];

                                // Вопросы к предложному дополнению
                                string qword = "на кого";
                                string answer = pn;

                                // На кого посмотрел брат?
                                WritePermutationsQA2(phrase, answer, qword, v, s);

                                // На кого пристально посмотрел брат?
                                WritePermutationsQA2(phrase, answer, qword, a, v, s);

                                // Вопросы к подлежащему
                                qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    // Кто пристально посмотрел на доктора?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, a, v2, pn);

                                    // Кто посмотрел на доктора?
                                    WritePermutationsQA2(phrase, answer, qword, v2, pn);
                                }

                                // Вопросы к объекту в предложном обстоятельстве:
                                // На кого посмотрел брат?
                                qword = TermsToString(gren, terms[3]) + " кого";
                                answer = TermsToString(gren, terms.Skip(3).Take(2));
                                WritePermutationsQA2(phrase, answer, qword, v, s);
                            }
                            #endregion Ткань хорошо держит форму


                            #region Дверца открывается влево.
                            if (footprint.Match("n,nom v adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // дверца
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // открывается
                                string a = TermsToString(gren, terms.Skip(2).Take(1)); // влево
                                var v_node = terms[1];

                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Куда открывается дверца?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, qword, v, s);
                                }


                                // Вопросы к подлежащему
                                qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    // Что открывается влево?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, a);
                                }
                            }
                            #endregion Дверца открывается влево.


                            #region Внутри находится карман.
                            if (footprint.Match("adv v n,nom"))
                            {
                                used = true;

                                string a = TermsToString(gren, terms.Take(1)); // внутри
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // находится
                                string s = TermsToString(gren, terms.Skip(2).Take(1)); // карман
                                var v_node = terms[1];

                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Где находится карман?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, qword, v, s);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[2]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что находится внутри?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Внутри находится карман.


                            #region Улица активно застраивается.
                            if (footprint.Match("n,nom adv v"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // улица
                                string a = TermsToString(gren, terms.Skip(1).Take(1)); // активно
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // застраивается
                                var v_node = terms[2];

                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Как застраивается улица?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, qword, v, s);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что застраивается активно?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Улица активно застраивается.


                            #region Мыло убивает запахи.
                            if (footprint.Match("n,nom v,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // мыло
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // убивает
                                string o = TermsToString(gren, terms.Skip(2).Take(1)); // запахи
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    qword = GetAccusObjectQuestion(footprint[2]);

                                    // Мыло убивает что?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что убивает запахи?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Мыло убивает запахи.


                            #region Заняли бойцы оборону.
                            if (footprint.Match("v,acc n,nom n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Skip(1).Take(1)); // бойцы
                                string v = TermsToString(gren, terms.Take(1)); // заняли
                                string o = TermsToString(gren, terms.Skip(2).Take(1)); // оборону
                                var v_node = terms[0];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    qword = GetAccusObjectQuestion(footprint[2]);

                                    // Бойцы заняли что?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Кто занял оборону?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Заняли бойцы оборону.


                            #region Оборону заняли бойцы.
                            if (footprint.Match("n,acc v,acc n,nom"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Skip(2).Take(1)); // бойцы
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // заняли
                                string o = TermsToString(gren, terms.Take(1)); // оборону
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    qword = GetAccusObjectQuestion(footprint[0]);

                                    // Бойцы заняли что?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);


                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[2]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Кто занял оборону?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Оборону заняли бойцы.


                            #region Мама удивилась вопросу.
                            if (footprint.Match("n,nom v,dat n,dat"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // мама
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // удивилась
                                string o = TermsToString(gren, terms.Skip(2).Take(1)); // вопросу
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    qword = GetDativeObjectQuestion(footprint[2]);

                                    // Мама удивилась чему?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Кто удивился вопросу?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Мама удивилась вопросу.


                            #region Дорога зимой чистится.
                            if (footprint.Match("n,nom n,instr v"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // дорога
                                string o = TermsToString(gren, terms.Skip(1).Take(1)); // зимой
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // чистится
                                var v_node = terms[2];

                                string qword = null;
                                string answer = null;

                                if (IsTimeNoun(o))
                                {
                                    // Дорога зимой чистится.
                                    qword = "когда";
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, qword, v, s); // Когда чистится дорога?
                                }
                                else if (IsGoodObject(o))
                                {
                                    // Вопросы к прямому дополнению
                                    qword = GetInstrObjectQuestion(footprint[1]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);
                                }

                                // Вопросы к подлежащему
                                qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    // Что чистится зимой?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, o);
                                }
                            }
                            #endregion Дорога зимой чистится.


                            #region Ручка регулируется кнопкой.
                            if (footprint.Match("n,nom v,instr n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // ручка
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // регулируется
                                string o = TermsToString(gren, terms.Skip(2).Take(1)); // кнопкой
                                var v_node = terms[1];

                                string qword = null;
                                string answer = null;

                                if (IsTimeNoun(o))
                                {
                                    // Ночью ударит мороз.
                                    qword = "когда";
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, qword, v, s); // Когда ударит мороз?
                                }
                                else if (IsGoodObject(o))
                                {
                                    // Вопросы к прямому дополнению
                                    qword = GetInstrObjectQuestion(footprint[2]);
                                    answer = o;

                                    // Ручка регулируется чем?
                                    WritePermutationsQA2(phrase, answer, s, v, qword);
                                }

                                // Вопросы к подлежащему
                                qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    // Что регулируется кнопкой?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, o);
                                }
                            }
                            #endregion Ручка регулируется кнопкой.


                            #region Ночью ударит мороз.
                            if (footprint.Match("n,instr v n,nom"))
                            {
                                used = true;

                                string o = TermsToString(gren, terms.Take(1)); // ночью
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // ударит
                                string s = TermsToString(gren, terms.Skip(2).Take(1)); // мороз
                                var v_node = terms[1];

                                string qword = null;
                                string answer = null;

                                if (IsTimeNoun(o))
                                {
                                    // Ночью ударит мороз.
                                    qword = "когда";
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, qword, v, s);
                                }
                                else if (IsGoodObject(o))
                                {
                                    // Прахом пошли усилия

                                    // Вопросы к прямому дополнению
                                    qword = GetInstrObjectQuestion(footprint[0]);

                                    // Усилия пошли чем?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);
                                }

                                // Вопросы к подлежащему
                                qword = GetSubjectQuestion(footprint[2]);

                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    // Что пошло прахом?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, o);
                                }
                            }
                            #endregion Прахом пошли усилия.


                            #region Бригада плотников ищет работу
                            if (footprint.Match("n,nom n,gen v,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // бригада плотников
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // ищет
                                string o = TermsToString(gren, terms.Skip(3).Take(1)); // работу
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению:
                                    // Бригада плотников ищет что?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что ищет работу?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Бригада плотников ищет работу


                            #region Занятия проводятся опытными тренерами
                            if (footprint.Match("n,nom v,instr adj,instr n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1));
                                string v = TermsToString(gren, terms.Skip(1).Take(1));
                                string o = TermsToString(gren, terms.Skip(2).Take(2));

                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к обстоятельственному дополнению
                                    // Занятия проводятся кем?
                                    qword = GetInstrObjectQuestion(footprint[3]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, o, v2, qword);
                                    }

                                    // Вопросы "какой? etc" к прямому дополнению
                                    // Какими тренерами проводятся занятия?
                                    qword = GetWhichQword4Instr(footprint[3]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        string o2 = qword + " " + TermsToString(gren, terms.Skip(3).Take(1)); // какими тренерами
                                        answer = TermsToString(gren, terms.Skip(2).Take(1)); // опытными
                                        WritePermutationsQA2(phrase, answer, o2, v, s);
                                    }
                                }
                            }
                            #endregion Занятия проводятся опытными тренерами


                            #region Ткань хорошо держит форму
                            if (footprint.Match("n,nom adv v,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // ткань
                                string v = TermsToString(gren, terms.Skip(1).Take(2)); // хорошо держит
                                string o = TermsToString(gren, terms.Skip(3).Take(1)); // форму
                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Ткань хорошо держит что?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s, v, qword);
                                    }


                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[0]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        string v2 = TermsToString(gren, terms.Skip(1).Take(1)) + " " + RebuildVerb2(gren, v_node, qword);

                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            // Что хорошо держит форму?
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v2, o);
                                        }
                                    }
                                }
                            }
                            #endregion Ткань хорошо держит форму


                            #region Быстро красит машина материю!
                            if (footprint.Match("adv v,acc n,nom n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Skip(2).Take(1)); // машина
                                string v = TermsToString(gren, terms.Take(2)); // быстро красит
                                string o = TermsToString(gren, terms.Skip(3).Take(1)); // ткань
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Машина быстро красит что?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[2]);
                                    string v2 = TermsToString(gren, terms.Take(1)) + " " + RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что быстро красит ткань?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Быстро красит машина материю!


                            #region Анастасия обрела самообладание быстро.
                            if (footprint.Match("n,nom v,acc n,acc adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // Анастасия
                                string v = TermsToString(gren, terms.Skip(3)) + " " + TermsToString(gren, terms.Skip(1).Take(1)); // быстро красит
                                string o = TermsToString(gren, terms.Skip(2).Take(1)); // самообладание
                                var v_node = terms[1];

                                if (IsGoodObject(o))
                                {
                                    string qword = null;
                                    string answer = null;

                                    // Вопросы к прямому дополнению
                                    // Анастасия быстро обрела что?
                                    qword = GetAccusObjectQuestion(footprint[2]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);


                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = TermsToString(gren, terms.Skip(3).Take(1)) + " " + RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Кто быстро обрел самообладание?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Анастасия обрела самообладание быстро.


                            #region Стекло не зарастает водорослями
                            if (footprint.Match("n,nom neg v,instr n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // стекло
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // зарастает
                                string o = TermsToString(gren, terms.Skip(3).Take(1)); // водорослями
                                var v_node = terms[2];

                                string answer = null;

                                // Стекло зарастает водорослями?
                                answer = "нет";
                                WritePermutationsQA2(phrase, answer, s, v, o);
                                //WritePermutationsQA2(phrase, answer, s, v+" ли", o);

                                // Вопросы к подлежащему
                                string qword = GetSubjectQuestion(footprint[0]);
                                string v0 = RebuildVerb2(gren, v_node, qword);
                                string v2 = "не " + v0;

                                if (!string.IsNullOrEmpty(v0))
                                {
                                    // Что не зарастает водорослями?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, o);
                                }

                                // Вопросы к обстоятельственному дополнению
                                qword = GetInstrObjectQuestion(footprint[3]);

                                v2 = "не " + v;

                                // Стекло не зарастает чем?
                                answer = o;
                                WritePermutationsQA2(phrase, answer, s, v2, qword);
                            }
                            #endregion Стекло не зарастает водорослями


                            #region Предложение не является публичной офертой.
                            if (footprint.Match("n,nom neg v,instr adj,instr n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // предложение
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // является
                                string o = TermsToString(gren, terms.Skip(3).Take(2)); // публичной офертой

                                string answer = null;

                                if (IsGoodObject(o))
                                {
                                    // Предложение является публичной офертой?
                                    answer = "нет";
                                    WritePermutationsQA2(phrase, answer, s, v, o);
                                    //WritePermutationsQA2(phrase, answer, s, v+" ли", o);

                                    // Вопросы к подлежащему
                                    string qword = GetSubjectQuestion(footprint[0]);

                                    var v_node = terms[2];
                                    string v0 = RebuildVerb2(gren, v_node, qword);
                                    string v2 = "не " + v0;

                                    if (!string.IsNullOrEmpty(v0))
                                    {
                                        // Что не является публичной офертой?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }

                                    // Вопросы к обстоятельственному дополнению
                                    qword = GetInstrObjectQuestion(footprint[3]);
                                    v2 = "не " + v;

                                    // Предложение не является чем?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v2, qword);
                                }
                            }
                            #endregion Предложение не является публичной офертой.


                            #region Скорость подачи регулируется инвертером
                            if (footprint.Match("n,nom n,gen v,instr n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // скорость подачи
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // регулируется
                                string o = TermsToString(gren, terms.Skip(3)); // инвертером
                                var v_node = terms[2];

                                string qword = "";
                                string answer = null;

                                // Вопросы к обстоятельственному дополнению в творительном падеже
                                qword = GetInstrObjectQuestion(footprint[3]);

                                if (IsGoodObject(o))
                                {
                                    // Скорость подачи регулируется чем?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);
                                }


                                // Вопрос к подлежащему:
                                // Что регулируется инвертором?
                                // ^^^
                                qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    // Что регулируется инвертером?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, o);
                                }
                            }
                            #endregion Скорость подачи регулируется инвертером


                            #region Машинка не нагревает воду
                            if (
                            footprint.Match("n,nom neg v,acc n") // Машинка не нагревает воду
                            || footprint.Match("n,nom n,acc neg v,acc") // Двигатель масло не расходует
                            || footprint.Match("n,gen n,nom neg v,acc") // Вложений машина не потребует
                            || footprint.Match("n,nom n,dat neg v,acc") // Паутина работе не мешает
                            || footprint.Match("n,dat neg v,acc n,nom") // Малышу не подошел размер
                               )
                            {
                                used = true;

                                string o = null;
                                string s = null;
                                string v = null;
                                SolarixGrammarEngineNET.SyntaxTreeNode obj_node = null;


                                if (footprint.Match("n,nom neg v n")) // Машинка не нагревает воду
                                {
                                    s = TermsToString(gren, terms.Take(1)); // машинка
                                    v = TermsToString(gren, terms.Skip(2).Take(1));
                                    obj_node = terms[3];
                                    o = TermsToString(gren, terms.Skip(3).Take(1)); // воду
                                }
                                else if (footprint.Match("n,nom n neg v")) // Двигатель масло не расходует
                                {
                                    s = TermsToString(gren, terms.Take(1)); // двигатель
                                    v = TermsToString(gren, terms.Skip(3).Take(1)); // расходует
                                    obj_node = terms[1];
                                    o = TermsToString(gren, terms.Skip(1).Take(1)); // масло
                                }
                                else if (footprint.Match("n,gen n,nom neg v")) // Вложений машина не потребует
                                {
                                    obj_node = terms[0];
                                    o = TermsToString(gren, terms.Take(1)); // вложений
                                    s = TermsToString(gren, terms.Skip(1).Take(1)); // машина
                                    v = TermsToString(gren, terms.Skip(3).Take(1)); // потребует
                                }
                                else if (footprint.Match("n,nom n,dat neg v")) // Паутина работе не мешает
                                {
                                    s = TermsToString(gren, terms.Take(1)); // паутина
                                    obj_node = terms[0];
                                    o = TermsToString(gren, terms.Skip(1).Take(1)); // работе
                                    v = TermsToString(gren, terms.Skip(3).Take(1)); // мешает
                                }
                                else if (footprint.Match("n,dat neg v n,nom")) // Малышу не подошел размер
                                {
                                    obj_node = terms[0];
                                    o = TermsToString(gren, terms.Take(1)); // малышу
                                    v = TermsToString(gren, terms.Skip(2).Take(1)); // подошел
                                    s = TermsToString(gren, terms.Skip(3).Take(1)); // размер
                                }

                                if (IsGoodObject(o))
                                {
                                    if (obj_node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.CASE_ru) == SolarixGrammarEngineNET.GrammarEngineAPI.GENITIVE_CASE_ru)
                                    {
                                        // Заменим на винительный падеж:
                                        // Парик не требует укладки ==> Парик требует укладку
                                        List<int> coords = new List<int>();
                                        List<int> states = new List<int>();

                                        foreach (var p in obj_node.GetPairs())
                                        {
                                            if (p.CoordID == SolarixGrammarEngineNET.GrammarEngineAPI.CASE_ru)
                                            {
                                                coords.Add(p.CoordID);
                                                states.Add(SolarixGrammarEngineNET.GrammarEngineAPI.ACCUSATIVE_CASE_ru);
                                            }
                                            else
                                            {
                                                coords.Add(p.CoordID);
                                                states.Add(p.StateID);
                                            }
                                        }


                                        List<string> fx = SolarixGrammarEngineNET.GrammarEngine.sol_GenerateWordformsFX(gren.GetEngineHandle(), obj_node.GetEntryID(), coords, states);
                                        if (fx.Count > 0)
                                        {
                                            o = fx[0].ToLower();
                                        }
                                        else
                                        {
                                            o = null;
                                        }
                                    }

                                    string answer = null;
                                    if (!string.IsNullOrEmpty(o))
                                    {
                                        used = true;

                                        // Машинка нагревает воду?
                                        answer = "нет";
                                        WritePermutationsQA2(phrase, answer, s, v, o);
                                        //WritePermutationsQA2(phrase, answer, s, v+" ли", o);
                                    }
                                }
                            }
                            #endregion Машинка не нагревает воду


                            #region Часы имеют оригинальное происхождение
                            // Часы имеют оригинальное происхождение
                            if (footprint.Match("n,nom v,acc adj,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1));
                                string v = TermsToString(gren, terms.Skip(1).Take(1));
                                string o = TermsToString(gren, terms.Skip(2).Take(2));
                                var v_node = terms[1];

                                string qword = null;
                                string answer = null;

                                if (IsGoodObject(o))
                                {
                                    // Вопросы к прямому дополнению
                                    // Часы имеют что?
                                    // Часы имеют кого?
                                    qword = GetAccusObjectQuestion(footprint[3]);
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что имеет оригинальное происхождение?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }

                                    // Вопросы "какой? etc" к прямому дополнению
                                    // Какое происхождение имеют часы?
                                    qword = GetWhichQword4Obj(footprint[3]);
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        string o2 = qword + " " + TermsToString(gren, terms.Skip(3).Take(1)); // происхождение
                                        answer = TermsToString(gren, terms.Skip(2).Take(1)); // оригинальное

                                        // Какое происхождение имеют часы?
                                        WritePermutationsQA2(phrase, answer, o2, v, s);
                                    }
                                }
                            }
                            #endregion Часы имеют оригинальное происхождение


                            #region Пластиковые окна закрываются ролставнями
                            // Пластиковые окна закрываются ролставнями
                            if (footprint.Match("adj,nom n,nom v n,instr"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2));
                                string v = TermsToString(gren, terms.Skip(2).Take(1));
                                string o = TermsToString(gren, terms.Skip(3));
                                var v_node = terms[2];

                                string qword = null;
                                string answer = null;

                                // Вопросы к обстоятельственному дополнению в творительном падеже
                                qword = GetInstrObjectQuestion(footprint[3]);

                                if (IsGoodObject(o))
                                {
                                    // Пластиковые окна закрываются чем?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);
                                }


                                // Вопрос к атрибуту подлежащего:
                                // Какие окна закрываются ставнями?
                                // ^^^^^
                                qword = "";
                                if (footprint[1].Match("sing"))
                                {
                                    int gender = terms[0].GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.GENDER_ru);
                                    switch (gender)
                                    {
                                        case SolarixGrammarEngineNET.GrammarEngineAPI.MASCULINE_GENDER_ru: qword = "какой"; break;
                                        case SolarixGrammarEngineNET.GrammarEngineAPI.FEMININE_GENDER_ru: qword = "какая"; break;
                                        case SolarixGrammarEngineNET.GrammarEngineAPI.NEUTRAL_GENDER_ru: qword = "какое"; break;
                                    }
                                }
                                else
                                {
                                    qword = "какие";
                                }

                                string s2 = qword + " " + TermsToString(gren, terms.Skip(1).Take(1));
                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Какие окна закрываются ставнями?
                                    answer = TermsToString(gren, terms.Take(1)); // пластиковые
                                    WritePermutationsQA2(phrase, answer, s2, v, o);
                                }


                                // Вопрос к подлежащему:
                                // Что закрывается ставнями?
                                // ^^^
                                qword = GetSubjectQuestion(footprint[1]);
                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    // Что закрывается ставнями?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, o);
                                }
                            }
                            #endregion Пластиковые окна закрываются ролставнями


                            #region Задняя дверь открывается вверх.
                            if (footprint.Match("adj,nom n,nom v adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // задняя дверь
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // открывается
                                string a = TermsToString(gren, terms.Skip(3)); // вверх
                                var v_node = terms[2];

                                // Вопросы к обстоятельству
                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Задняя дверь открывается куда?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопрос к атрибуту подлежащего:
                                    // Какая дверь открывается вверх?
                                    // ^^^^^
                                    qword = GetWhichQword4Subject(footprint[1]);
                                    string s2 = qword + " " + TermsToString(gren, terms.Skip(1).Take(1));
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = TermsToString(gren, terms.Take(1)); // задняя

                                        // Какая дверь открывается вверх?
                                        WritePermutationsQA2(phrase, answer, s2, v, a);
                                    }


                                    // Вопрос к подлежащему:
                                    // Что открывается вверх?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);
                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что открывается вверх?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Задняя дверь открывается вверх.


                            #region Задняя дверь вверх открывается.
                            if (footprint.Match("adj,nom n,nom adv v"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // задняя дверь
                                string a = TermsToString(gren, terms.Skip(2).Take(1)); // вверх
                                string v = TermsToString(gren, terms.Skip(3).Take(1)); // открывается
                                var v_node = terms[3];

                                // Вопросы к обстоятельству
                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Задняя дверь открывается куда?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);


                                    // Вопрос к атрибуту подлежащего:
                                    // Какая дверь открывается вверх?
                                    // ^^^^^
                                    qword = GetWhichQword4Subject(footprint[1]);
                                    string s2 = qword + " " + TermsToString(gren, terms.Skip(1).Take(1));
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = TermsToString(gren, terms.Take(1)); // задняя

                                        // Какая дверь открывается вверх?
                                        WritePermutationsQA2(phrase, answer, s2, v, a);
                                    }


                                    // Вопрос к подлежащему:
                                    // Что открывается вверх?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что открывается вверх?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Задняя дверь вверх открывается.


                            #region Вверх открывается задняя дверь.
                            if (footprint.Match("adv v adj,nom n,nom"))
                            {
                                used = true;

                                string a = TermsToString(gren, terms.Take(1)); // вверх
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // открывается
                                string s = TermsToString(gren, terms.Skip(2).Take(2)); // задняя дверь
                                var v_node = terms[1];

                                // Вопросы к обстоятельству
                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Задняя дверь открывается куда?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);


                                    // Вопрос к атрибуту подлежащего:
                                    // Какая дверь открывается вверх?
                                    // ^^^^^
                                    qword = GetWhichQword4Subject(footprint[3]);
                                    string s2 = qword + " " + TermsToString(gren, terms.Skip(3).Take(1)); // Какая дверь
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = TermsToString(gren, terms.Skip(2).Take(1)); // задняя

                                        // Какая дверь открывается вверх?
                                        WritePermutationsQA2(phrase, answer, s2, v, a);
                                    }


                                    // Вопрос к подлежащему:
                                    // Что открывается вверх?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[3]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что открывается вверх?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Задняя дверь вверх открывается.


                            #region открывается вверх задняя дверь.
                            if (footprint.Match("v adv adj,nom n,nom"))
                            {
                                used = true;

                                string v = TermsToString(gren, terms.Take(1)); // открывается
                                string a = TermsToString(gren, terms.Skip(1).Take(1)); // вверх
                                string s = TermsToString(gren, terms.Skip(2).Take(2)); // задняя дверь
                                var v_node = terms[0];

                                // Вопросы к обстоятельству
                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Задняя дверь открывается куда?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопрос к атрибуту подлежащего:
                                    // Какая дверь открывается вверх?
                                    // ^^^^^
                                    qword = GetWhichQword4Subject(footprint[3]);
                                    string s2 = qword + " " + TermsToString(gren, terms.Skip(3).Take(1)); // Какая дверь
                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = TermsToString(gren, terms.Skip(2).Take(1)); // задняя

                                        // Какая дверь открывается вверх?
                                        WritePermutationsQA2(phrase, answer, s2, v, a);
                                    }


                                    // Вопрос к подлежащему:
                                    // Что открывается вверх?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[3]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что открывается вверх?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Задняя дверь вверх открывается.


                            #region Панель управления откидывается вверх.
                            if (footprint.Match("n,nom n,gen v adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // панель управления
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // откидывается
                                string a = TermsToString(gren, terms.Skip(3)); // вверх
                                var v_node = terms[2];

                                // Вопросы к обстоятельству
                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Панель управления откидывается куда?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);

                                    // Вопрос к подлежащему:
                                    // Что откидывается вверх?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что откидывается вверх?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Панель управления откидывается вверх.


                            #region Вверх откидывается панель управления
                            if (footprint.Match("adv v n,nom n,gen"))
                            {
                                used = true;

                                string a = TermsToString(gren, terms.Take(1)); // вверх
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // откидывается
                                string s = TermsToString(gren, terms.Skip(2).Take(2)); // панель управления
                                var v_node = terms[1];

                                // Вопросы к обстоятельству
                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Панель управления откидывается куда?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);


                                    // Вопрос к подлежащему:
                                    // Что откидывается вверх?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[2]);

                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что откидывается вверх?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Вверх откидывается панель управления


                            #region Панель управления вверх откидывается.
                            if (footprint.Match("n,nom n,gen adv v"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // панель управления
                                string a = TermsToString(gren, terms.Skip(2).Take(1)); // вверх
                                string v = TermsToString(gren, terms.Skip(3).Take(1)); // откидывается
                                var v_node = terms[3];

                                // Вопросы к обстоятельству
                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Панель управления откидывается куда?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);


                                    // Вопрос к подлежащему:
                                    // Что откидывается вверх?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[0]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что откидывается вверх?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Панель управления вверх откидывается.


                            #region Вверх панель управления откидывается
                            if (footprint.Match("adv n,nom n,gen v"))
                            {
                                used = true;

                                string a = TermsToString(gren, terms.Take(1)); // вверх
                                string s = TermsToString(gren, terms.Skip(1).Take(2)); // панель управления
                                string v = TermsToString(gren, terms.Skip(3).Take(1)); // откидывается
                                var v_node = terms[3];

                                // Вопросы к обстоятельству
                                string qword = GetQuestionWordForAdverb(a);
                                string answer = null;

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Панель управления откидывается куда?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);


                                    // Вопрос к подлежащему:
                                    // Что откидывается вверх?
                                    // ^^^
                                    qword = GetSubjectQuestion(footprint[1]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Что откидывается вверх?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, a);
                                    }
                                }
                            }
                            #endregion Вверх панель управления откидывается


                            #region Собственник заключает договор аренды
                            // Собственник заключает договор аренды
                            if (footprint.Match("n,nom v n,acc n,gen"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // собственник
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // заключает
                                string o = TermsToString(gren, terms.Skip(2)); // договор аренды
                                var v_node = terms[1];

                                string qword = null;
                                string answer = null;

                                // Вопрос к прямому дополнению
                                qword = GetAccusObjectQuestion(footprint[2]);

                                if (IsGoodObject(o))
                                {
                                    // Собственник заключает что?
                                    answer = o;
                                    WritePermutationsQA2(phrase, answer, s, v, qword);
                                }

                                // Вопросы к подлежащему
                                qword = GetSubjectQuestion(footprint[0]);
                                string v2 = RebuildVerb2(gren, v_node, qword);

                                if (!string.IsNullOrEmpty(v2))
                                {
                                    // Кто заключает договор аренды?
                                    answer = s;
                                    WritePermutationsQA2(phrase, answer, qword, v2, o);
                                }
                            }
                            #endregion Собственник заключает договор аренды


                            #region Высокие волны заливали пляж
                            if (footprint.Match("adj,nom n,nom v,acc n,acc"))
                            {
                                used = true;

                                // Высокие волны заливали пляж
                                // Высокая волна залила пляж

                                string s = TermsToString(gren, terms.Take(2));
                                string v = TermsToString(gren, terms.Skip(2).Take(1));
                                string o = TermsToString(gren, terms.Skip(3).Take(1));

                                var v_node = terms[2];

                                if (IsGoodObject(o))
                                {
                                    // Вопросы к атрибуту подлежащего:
                                    // Какие волны заливали пляж?
                                    string qword = GetWhichQword4Subject(footprint[1]);
                                    string answer = "";

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        used = true;

                                        string s2 = qword + " " + TermsToString(gren, terms.Skip(1).Take(1));
                                        answer = terms[0].GetWord();

                                        // Какой кот ищет подружку?
                                        WritePermutationsQA2(phrase, answer, s2, v, o);
                                    }


                                    // Вопросы к прямому дополнению.
                                    qword = GetAccusObjectQuestion(footprint[3]);

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        // Что снимет семейная пара?
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, qword, v, s);
                                    }


                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[1]);

                                    // Иногда требуется изменить число для глагола:
                                    // Глухие удары раскалывают тишину --> Что раскалывает тишину?
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Кто ищет подружку?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Высокие волны заливали пляж


                            #region Сетку обслуживает опытный человек
                            if (footprint.Match("n,acc v,acc adj,nom n,nom"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Skip(2).Take(2)); // опытный человек
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // обслуживает
                                string o = TermsToString(gren, terms.Take(1)); // сетку
                                var v_node = terms[1]; // обслуживает

                                string answer = "";

                                // Сетку обслуживает опытный человек
                                string qword = GetWhichQword4Subject(footprint[2]);

                                if (IsGoodObject(o))
                                {
                                    // Вопросы к атрибуту подлежащего

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        used = true;

                                        answer = TermsToString(gren, terms.Skip(2).Take(1)); // опытный
                                        s = qword + " " + TermsToString(gren, terms.Skip(3).Take(1)); // человек

                                        // Какой человек обслуживает сетку?
                                        WritePermutationsQA2(phrase, answer, s, v, o);
                                    }


                                    // Вопросы к прямому дополнению.
                                    qword = GetAccusObjectQuestion(footprint[0]);

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        s = TermsToString(gren, terms.Skip(2).Take(2)); // опытный человек

                                        // Опытный человек обслуживает что?
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s, v, qword);
                                    }


                                    // Вопросы к подлежащему
                                    qword = GetSubjectQuestion(footprint[3]);
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        // Кто обслуживает сетку?
                                        answer = s;
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }
                                }
                            }
                            #endregion Сетку обслуживает опытный человек


                            #region Массажные головки имеют встроенный нагрев
                            if (footprint.Match("adj,nom n,nom v,acc adj,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(2)); // массажные головки
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // имеют
                                string o = TermsToString(gren, terms.Skip(3).Take(2)); // встроенный нагрев

                                string sn = TermsToString(gren, terms.Skip(1).Take(1)); // головки
                                string on = TermsToString(gren, terms.Skip(4).Take(1)); // нагрев

                                if (IsGoodObject(o))
                                {
                                    // Вопросы к атрибуту прямого дополнения
                                    // Массажные головки имеют встроенный нагрев.
                                    //                         ^^^^^^^^^^
                                    // Какой нагрев имеют массажные головки?
                                    // ^^^^^

                                    string qword = "";
                                    var ft = footprint[3];
                                    if (ft.Match("anim,sing,masc"))
                                    {
                                        qword = "какого";
                                    }
                                    else if (ft.Match("anim,sing,fem"))
                                    {
                                        qword = "какую";
                                    }
                                    else if (ft.Match("anim,sing,neut"))
                                    {
                                        qword = "какое";
                                    }
                                    else if (ft.Match("anim,pl"))
                                    {
                                        qword = "каких";
                                    }
                                    else if (ft.Match("inanim,sing,masc"))
                                    {
                                        qword = "какой";
                                    }
                                    else if (ft.Match("inanim,sing,fem"))
                                    {
                                        qword = "какую";
                                    }
                                    else if (ft.Match("inanim,sing,neut"))
                                    {
                                        qword = "какое";
                                    }
                                    else if (ft.Match("inanim,pl"))
                                    {
                                        qword = "какие";
                                    }

                                    string question = "";
                                    string answer = null;

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        used = true;

                                        string o2 = qword + " " + on;

                                        answer = TermsToString(gren, terms.Skip(3).Take(1)); // встроенный
                                        WritePermutationsQA2(phrase, answer, s, v, o2);
                                    }


                                    // Вопрос к атрибуту подлежащего:
                                    // Какие головки имеют встроенный нагрев?
                                    qword = GetWhichQword4Subject(footprint[1]);

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        used = true;

                                        string s2 = qword + " " + sn;

                                        answer = TermsToString(gren, terms.Take(1)); // массажные
                                        WritePermutationsQA2(phrase, answer, s2, v, o);
                                    }


                                    // Вопросы к прямому дополнению
                                    qword = GetAccusObjectQuestion(footprint[4]);

                                    if (!string.IsNullOrEmpty(qword))
                                    {
                                        answer = o;
                                        WritePermutationsQA2(phrase, answer, s, v, qword);
                                    }


                                    // Вопросы к подлежащему
                                    // Массажная головка имеет встроенный нагрев
                                    // Что имеет встроенный нагрев?
                                    qword = GetSubjectQuestion(footprint[1]);

                                    var v_node = terms[2];
                                    string v2 = RebuildVerb2(gren, v_node, qword);

                                    if (!string.IsNullOrEmpty(v2))
                                    {
                                        answer = s;

                                        // Что имеет встроенный нагрев?
                                        WritePermutationsQA2(phrase, answer, qword, v2, o);
                                    }

                                }
                            }
                            #endregion Массажные головки имеют встроенный нагрев


                            #region Кот повел ребят вверх.
                            if (footprint.Match("n,nom v,acc n,acc adv"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // кот
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // повел
                                string o = TermsToString(gren, terms.Skip(2).Take(1)); // ребят
                                string a = TermsToString(gren, terms.Skip(3).Take(1)); // вверх

                                var v_node = terms[1];
                                string qword = null;
                                string answer = null;

                                // Вопросы к обстоятельству
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Кот куда повел ребят?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, qword, v, o);

                                    if (IsGoodObject(o))
                                    {
                                        // Вопросы к прямому дополнению.
                                        qword = GetAccusObjectQuestion(footprint[2]);
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            // Кого повел вверх кот?
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, qword, v, a, s);
                                        }


                                        // Вопросы к подлежащему
                                        qword = GetSubjectQuestion(footprint[0]);

                                        // Иногда требуется изменить число для глагола:
                                        // Глухие удары раскалывают тишину --> Что раскалывает тишину?
                                        string v2 = RebuildVerb2(gren, v_node, qword);

                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            // Кто повел ребят вверх?
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v2, o, a);
                                        }
                                    }
                                }
                            }
                            #endregion Кот повел ребят вверх.


                            #region Кот вверх повел ребят.
                            if (footprint.Match("n,nom adv v,acc n,acc"))
                            {
                                used = true;

                                string s = TermsToString(gren, terms.Take(1)); // кот
                                string a = TermsToString(gren, terms.Skip(1).Take(1)); // вверх
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // повел
                                string o = TermsToString(gren, terms.Skip(3).Take(1)); // ребят

                                var v_node = terms[2];
                                string qword = null;
                                string answer = null;

                                // Вопросы к обстоятельству
                                qword = GetQuestionWordForAdverb(a);
                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Кот куда повел ребят?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, qword, v, o);

                                    if (IsGoodObject(o))
                                    {
                                        // Вопросы к прямому дополнению.
                                        qword = GetAccusObjectQuestion(footprint[3]); // ребят
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            // Кого повел вверх кот?
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, qword, v, a, s);
                                        }


                                        // Вопросы к подлежащему
                                        qword = GetSubjectQuestion(footprint[0]);

                                        // Иногда требуется изменить число для глагола:
                                        // Глухие удары раскалывают тишину --> Что раскалывает тишину?
                                        string v2 = RebuildVerb2(gren, v_node, qword);

                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            // Кто повел ребят вверх?
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v2, o, a);
                                        }
                                    }
                                }
                            }
                            #endregion Кот вверх повел ребят.


                            #region Вверх кот повел ребят.
                            if (footprint.Match("adv n,nom v,acc n,acc"))
                            {
                                used = true;

                                string a = TermsToString(gren, terms.Take(1)); // вверх
                                string s = TermsToString(gren, terms.Skip(1).Take(1)); // кот
                                string v = TermsToString(gren, terms.Skip(2).Take(1)); // повел
                                string o = TermsToString(gren, terms.Skip(3).Take(1)); // ребят

                                var v_node = terms[2];
                                string qword = null;
                                string answer = null;

                                // Вопросы к обстоятельству
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Кот куда повел ребят?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, qword, v, o);

                                    if (IsGoodObject(o))
                                    {
                                        // Вопросы к прямому дополнению.
                                        qword = GetAccusObjectQuestion(footprint[3]); // ребят
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            // Кого повел вверх кот?
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, qword, v, a, s);
                                        }


                                        // Вопросы к подлежащему
                                        qword = GetSubjectQuestion(footprint[1]);

                                        // Иногда требуется изменить число для глагола:
                                        // Глухие удары раскалывают тишину --> Что раскалывает тишину?
                                        string v2 = RebuildVerb2(gren, v_node, qword);

                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            // Кто повел ребят вверх?
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v2, o, a);
                                        }
                                    }
                                }
                            }
                            #endregion Вверх кот повел ребят.


                            #region Вверх повел ребят кот.
                            if (footprint.Match("adv v,acc n,nom n,acc"))
                            {
                                used = true;

                                string a = TermsToString(gren, terms.Take(1)); // вверх
                                string v = TermsToString(gren, terms.Skip(1).Take(1)); // повел
                                string o = TermsToString(gren, terms.Skip(2).Take(1)); // ребят
                                string s = TermsToString(gren, terms.Skip(3).Take(1)); // кот

                                var v_node = terms[1];
                                string qword = null;
                                string answer = null;

                                // Вопросы к обстоятельству
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Кот куда повел ребят?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, qword, v, o);

                                    if (IsGoodObject(o))
                                    {
                                        // Вопросы к прямому дополнению.
                                        qword = GetAccusObjectQuestion(footprint[2]);
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            // Кого повел вверх кот?
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, qword, v, a, s);
                                        }


                                        // Вопросы к подлежащему
                                        qword = GetSubjectQuestion(footprint[3]);

                                        // Иногда требуется изменить число для глагола:
                                        // Глухие удары раскалывают тишину --> Что раскалывает тишину?
                                        string v2 = RebuildVerb2(gren, v_node, qword);

                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            // Кто повел ребят вверх?
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v2, o, a);
                                        }
                                    }
                                }
                            }
                            #endregion Вверх повел ребят кот.


                            #region Ребят вверх кот повел.
                            if (footprint.Match("n,acc adv n,nom v,acc"))
                            {
                                used = true;

                                string o = TermsToString(gren, terms.Take(1)); // ребят
                                string a = TermsToString(gren, terms.Skip(1).Take(1)); // вверх
                                string s = TermsToString(gren, terms.Skip(2).Take(1)); // кот
                                string v = TermsToString(gren, terms.Skip(3).Take(1)); // повел

                                var v_node = terms[3];
                                string qword = null;
                                string answer = null;

                                // Вопросы к обстоятельству
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Кот куда повел ребят?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, qword, v, o);

                                    if (IsGoodObject(o))
                                    {
                                        // Вопросы к прямому дополнению.
                                        qword = GetAccusObjectQuestion(footprint[0]); // ребят

                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            // Кого повел вверх кот?
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, qword, v, a, s);
                                        }


                                        // Вопросы к подлежащему
                                        qword = GetSubjectQuestion(footprint[2]);

                                        // Иногда требуется изменить число для глагола:
                                        // Глухие удары раскалывают тишину --> Что раскалывает тишину?
                                        string v2 = RebuildVerb2(gren, v_node, qword);

                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            // Кто повел ребят вверх?
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v2, o, a);
                                        }
                                    }
                                }
                            }
                            #endregion Ребят вверх кот повел.


                            #region Ребят кот вверх повел.
                            if (footprint.Match("n,acc n,nom adv v,acc"))
                            {
                                used = true;

                                string o = TermsToString(gren, terms.Take(1)); // ребят
                                string s = TermsToString(gren, terms.Skip(1).Take(1)); // кот
                                string a = TermsToString(gren, terms.Skip(2).Take(1)); // вверх
                                string v = TermsToString(gren, terms.Skip(3).Take(1)); // повел

                                var v_node = terms[3];
                                string qword = null;
                                string answer = null;

                                // Вопросы к обстоятельству
                                qword = GetQuestionWordForAdverb(a);

                                if (!string.IsNullOrEmpty(qword))
                                {
                                    // Кот куда повел ребят?
                                    answer = a;
                                    WritePermutationsQA2(phrase, answer, s, qword, v, o);

                                    if (IsGoodObject(o))
                                    {
                                        // Вопросы к прямому дополнению.
                                        qword = GetAccusObjectQuestion(footprint[0]); // ребят
                                        if (!string.IsNullOrEmpty(qword))
                                        {
                                            // Кого повел вверх кот?
                                            answer = o;
                                            WritePermutationsQA2(phrase, answer, qword, v, a, s);
                                        }


                                        // Вопросы к подлежащему
                                        qword = GetSubjectQuestion(footprint[1]);

                                        // Иногда требуется изменить число для глагола:
                                        // Глухие удары раскалывают тишину --> Что раскалывает тишину?
                                        string v2 = RebuildVerb2(gren, v_node, qword);

                                        if (!string.IsNullOrEmpty(v2))
                                        {
                                            // Кто повел ребят вверх?
                                            answer = s;
                                            WritePermutationsQA2(phrase, answer, qword, v2, o, a);
                                        }
                                    }
                                }
                            }
                            #endregion Ребят вверх кот повел.

                        }
                    }
                }
            }

            if (!used)
            {
                wrt_skipped.WriteLine("{0}", phrase);
                wrt_skipped.Flush();
            }
        }

        wrt_samples.Flush();

        Console.Write("{0} processed, {1} samples generated\r", nb_processed, nb_samples);

        return;
    }


    static HashSet<Int64> sample_hashes = new HashSet<Int64>();
    static MD5 md5 = MD5.Create();

    static string NormalizeSample(string str)
    {
        return str.ToLower();
    }

    static bool IsUniqueSample(string str)
    {
        byte[] hash = md5.ComputeHash(System.Text.Encoding.UTF8.GetBytes(str.ToLower()));
        Int64 ihash1 = BitConverter.ToInt64(hash, 0);
        Int64 ihash2 = BitConverter.ToInt64(hash, 8);
        Int64 ihash = ihash1 ^ ihash2;

        if (!sample_hashes.Contains(ihash))
        {
            sample_hashes.Add(ihash);
            return true;
        }
        else
        {
            return false;
        }
    }


    static bool IsGoodSent( string phrase )
    {
        string stops = "как ни в чем не бывало|тем  не менее";
        foreach( string s in stops.Split('|'))
        {
            if (phrase.Contains(s))
                return false;
        }

        return true;
    }

    static void Main(string[] args)
    {
        string result_folder = @"f:\tmp";
        List<string> input_filenames = new List<string>();

        int MAX_SAMPLE = int.MaxValue;
        int MAX_LEN = int.MaxValue;
        string dictionary_xml = "dictionary.xml";
        bool OnlyNegations = false;
        string rx_filter = null;

        #region Command_Line_Options
        for (int i = 0; i < args.Length; ++i)
        {
            if (args[i] == "-input")
            {
                input_filenames.Add(args[i + 1]);
                i++;
            }
            else if (args[i] == "-output")
            {
                result_folder = args[i + 1];
                i++;
            }
            else if (args[i] == "-max_samples")
            {
                MAX_SAMPLE = int.Parse(args[i + 1]);
                i++;
            }
            else if (args[i] == "-max_len")
            {
                MAX_LEN = int.Parse(args[i + 1]);
                i++;
            }
            else if (args[i] == "-only_neg")
            {
                OnlyNegations = true;
            }
            else if (args[i] == "-dict")
            {
                dictionary_xml = args[i + 1];
                i++;
            }
            else if (args[i] == "-rx")
            {
                rx_filter = args[i + 1].Trim();
                i++;
            }
            else
            {
                throw new ApplicationException(string.Format("Unknown option {0}", args[i]));
            }
        }
        #endregion Command_Line_Options

        wrt_samples = new System.IO.StreamWriter(System.IO.Path.Combine(result_folder, "premise_question_answer.txt"));
        wrt_skipped = new System.IO.StreamWriter(System.IO.Path.Combine(result_folder, "skipped.txt"));


        // Загружаем грамматический словарь
        Console.WriteLine("Loading dictionary {0}", dictionary_xml);
        SolarixGrammarEngineNET.GrammarEngine2 gren = new SolarixGrammarEngineNET.GrammarEngine2();
        gren.Load(dictionary_xml, true);

        #region Processing_All_Files
        foreach (string mask in input_filenames)
        {
            string[] files = null;
            if (System.IO.Directory.Exists(mask))
            {
                files = System.IO.Directory.GetFiles(mask, "*.txt");
            }
            else if (mask.IndexOfAny("*?".ToCharArray()) != -1)
            {
                files = System.IO.Directory.GetFiles(System.IO.Path.GetDirectoryName(mask), System.IO.Path.GetFileName(mask));
            }
            else
            {
                files = new string[1] { mask };
            }

            Console.WriteLine("Number of input files={0}", files.Length);

            foreach (string file in files)
            {
                if (sample_count >= MAX_SAMPLE)
                    break;

                Console.WriteLine("Processing {0}...", file);

                using (System.IO.StreamReader rdr = new System.IO.StreamReader(file))
                {
                    while (!rdr.EndOfStream && sample_count < MAX_SAMPLE)
                    {
                        string sent = rdr.ReadLine();
                        if (sent == null) break;
                        sent = sent.Trim();
                        sample_count++;

                        sent = sent.Replace("  ", " ");

                        string lsent = sent.ToLower();
                        if (!IsGoodSent(sent))
                            continue;

                        if (OnlyNegations)
                        {
                            bool has_neg = false;

                            if (!lsent.Contains("тем не менее"))
                            {
                                if (lsent.StartsWith("не ") || lsent.Contains(" не "))
                                {
                                    has_neg = true;
                                }
                            }

                            if (!has_neg)
                            {
                                continue;
                            }
                        }

                        if (!string.IsNullOrEmpty(rx_filter))
                        {
                            var mx = System.Text.RegularExpressions.Regex.Match(sent, rx_filter, System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                            if (!mx.Success)
                            {
                                continue;
                            }
                        }

                        try
                        {
                            ProcessSentence(sent, gren, MAX_LEN);
                        }
                        catch (Exception ex)
                        {
                            Console.Write("ERROR\nSentence={0}\nError={1}", sent, ex.Message);
                        }
                    }
                }
            }
        }
        #endregion Processing_All_Files

        wrt_samples.Close();
        wrt_skipped.Close();

        return;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class GeneratePersonChangeDataset
{
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


    static int GetPOS(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode node)
    {
        int id_entry = node.GetEntryID();
        int pos_id = gren.GetEntryClass(id_entry);
        return pos_id;
    }

    static bool IsPronoun_1s_nom(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode node)
    {
        return GetPOS(gren, node) == SolarixGrammarEngineNET.GrammarEngineAPI.PRONOUN_ru &&
            node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru) == SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_1_ru &&
            node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.CASE_ru) == SolarixGrammarEngineNET.GrammarEngineAPI.NOMINATIVE_CASE_ru &&
            node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru) == SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru;
    }

    static bool IsVerb_1s(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode node)
    {
        if (GetPOS(gren, node) == SolarixGrammarEngineNET.GrammarEngineAPI.VERB_ru)
        {
            if (node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_ru) == SolarixGrammarEngineNET.GrammarEngineAPI.PERSON_1_ru &&
                node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.NUMBER_ru) == SolarixGrammarEngineNET.GrammarEngineAPI.SINGULAR_NUMBER_ru &&
                node.GetCoordState(SolarixGrammarEngineNET.GrammarEngineAPI.VERB_FORM_ru) == SolarixGrammarEngineNET.GrammarEngineAPI.VB_INF_ru)
            {
                return true;
            }
        }

        return false;
    }


    static string ChangePronounTo(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode node, string to_person)
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


    static string ChangeVerbTo(SolarixGrammarEngineNET.GrammarEngine2 gren, SolarixGrammarEngineNET.SyntaxTreeNode node, string to_person)
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




    static int Main(string[] args)
    {
        List<string> input_files = new List<string>();
        string output_file = null;
        string dictionary_xml = "";
        string from_person = "";
        string to_person = "";

        #region Command_Line_Options
        for (int i = 0; i < args.Length; ++i)
        {
            if (args[i] == "-input_file")
            {
                input_files.Add(args[i + 1]);
                i++;
            }
            else if (args[i] == "-output_file")
            {
                output_file = args[i + 1];
                i++;
            }
            else if (args[i] == "-dict")
            {
                dictionary_xml = args[i + 1];
                i++;
            }
            else if (args[i] == "-from_person")
            {
                from_person = args[i + 1];
                i++;
            }
            else if (args[i] == "-to_person")
            {
                to_person = args[i + 1];
                i++;
            }
            else
            {
                throw new ApplicationException(string.Format("Unknown option {0}", args[i]));
            }
        }

        if (string.IsNullOrEmpty(from_person))
        {
            Console.WriteLine("'from_person' parameter can not be empty");
            return 1;
        }

        if (string.IsNullOrEmpty(to_person))
        {
            Console.WriteLine("'to_person' parameter can not be empty");
            return 1;
        }
        #endregion Command_Line_Options



        // Загружаем грамматический словарь
        Console.WriteLine("Loading dictionary {0}", dictionary_xml);
        SolarixGrammarEngineNET.GrammarEngine2 gren = new SolarixGrammarEngineNET.GrammarEngine2();
        gren.Load(dictionary_xml, true);


        int id_language = SolarixGrammarEngineNET.GrammarEngineAPI.RUSSIAN_LANGUAGE;
        SolarixGrammarEngineNET.GrammarEngine.MorphologyFlags morph_flags = SolarixGrammarEngineNET.GrammarEngine.MorphologyFlags.SOL_GREN_COMPLETE_ONLY | SolarixGrammarEngineNET.GrammarEngine.MorphologyFlags.SOL_GREN_MODEL;
        SolarixGrammarEngineNET.GrammarEngine.SyntaxFlags syntax_flags = SolarixGrammarEngineNET.GrammarEngine.SyntaxFlags.DEFAULT;
        int MaxAlt = 40;
        int constraints = 600000 | (MaxAlt << 22);

        using (System.IO.StreamWriter wrt = new System.IO.StreamWriter(output_file))
        {
            int nb_samples = 0;
            foreach (string input_path in input_files)
            {
                Console.WriteLine("Processing {0}", input_path);

                using (System.IO.StreamReader rdr = new System.IO.StreamReader(input_path))
                {
                    while (!rdr.EndOfStream)
                    {
                        string line0 = rdr.ReadLine();
                        if (line0 == null) break;

                        string line = line0.Trim();
                        string phrase2 = line;

                        using (SolarixGrammarEngineNET.AnalysisResults linkages = gren.AnalyzeSyntax(phrase2, id_language, morph_flags, syntax_flags, constraints))
                        {
                            if (linkages.Count == 3)
                            {
                                SolarixGrammarEngineNET.SyntaxTreeNode root = linkages[1];
                                List<SolarixGrammarEngineNET.SyntaxTreeNode> terms = GetTerms(root).OrderBy(z => z.GetWordPosition()).ToList();

                                if (from_person == "1s")
                                {
                                    // Ищем подлежащее-местоимение "я" или проверяем, что глагол стоит в первом лице.
                                    bool is_good_sample = false;
                                    if (IsVerb_1s(gren, root))
                                    {
                                        is_good_sample = true;
                                    }

                                    if (!is_good_sample)
                                    {
                                        for (int ichild = 0; ichild < root.leafs.Count; ++ichild)
                                        {
                                            if (root.GetLinkType(ichild) == SolarixGrammarEngineNET.GrammarEngineAPI.SUBJECT_link)
                                            {
                                                SolarixGrammarEngineNET.SyntaxTreeNode sbj = root.leafs[ichild];
                                                if (IsPronoun_1s_nom(gren, sbj))
                                                {
                                                    is_good_sample = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }

                                    if (is_good_sample)
                                    {
                                        // Не должно быть местоимений в других падежах, чтобы не получалось:
                                        // Я тебя съем !	ты тебя съешь !
                                        foreach( var term in terms )
                                        {
                                            if( GetPOS(gren,term)==SolarixGrammarEngineNET.GrammarEngineAPI.PRONOUN_ru && !IsPronoun_1s_nom(gren, term) )
                                            {
                                                is_good_sample = false;
                                                break;
                                            }
                                        }
                                    }

                                    if (is_good_sample)
                                    {
                                        List<string> src_words = new List<string>();
                                        List<string> res_words = new List<string>();
                                        foreach (var term in terms)
                                        {
                                            src_words.Add(term.GetWord());

                                            if (IsPronoun_1s_nom(gren, term))
                                            {
                                                string new_word = ChangePronounTo(gren, term, to_person);
                                                res_words.Add(new_word);
                                            }
                                            else if (IsVerb_1s(gren, term))
                                            {
                                                string new_word = ChangeVerbTo(gren, term, to_person);
                                                res_words.Add(new_word);
                                            }
                                            else
                                            {
                                                res_words.Add(term.GetWord());
                                            }
                                        }


                                        int nb_empty = res_words.Count(z => string.IsNullOrEmpty(z));
                                        if (nb_empty == 0)
                                        {
                                            string src_str = string.Join(" ", src_words);
                                            string res_str = string.Join(" ", res_words);
                                            wrt.WriteLine("{0}\t{1}", src_str, res_str);
                                            wrt.Flush();
                                            nb_samples++;
                                            if( (nb_samples%10)==0 )
                                            {
                                                Console.Write("{0} samples stored\r", nb_samples);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Console.WriteLine();
            }
        }


        return 0;
    }
}

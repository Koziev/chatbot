using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Security.Cryptography;


public static class SolarixExtender
{
    public static bool IsPartOfSpeech(this SolarixGrammarEngineNET.SyntaxTreeNode node, SolarixGrammarEngineNET.GrammarEngine2 gren, int part_of_speech)
    {
        int p = gren.GetEntryClass(node.GetEntryID());
        return p == part_of_speech;
    }
}


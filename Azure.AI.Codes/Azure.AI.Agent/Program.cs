using Azure;
using Azure.AI.Agents.Persistent;
using Azure.Identity;
using Microsoft.Extensions.Configuration;
using System.Diagnostics;

// Page: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart?pivots=programming-language-csharp

IConfigurationRoot configuration = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
    .Build();

// See https://aka.ms/new-console-template for more information
Console.ReadLine();
Console.WriteLine("Hello, World!");

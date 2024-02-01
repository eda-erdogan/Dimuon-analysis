#include "TH1.h"
#include "TFile.h"
#include <iostream>

void comparehistograms() {
  TFile *file1 = new TFile("dimuonoutput.root"); //dimuon_analysis dosyası 
  TFile *file2 = new TFile("dimuon_mass2.root"); //tut dosyası

  TH1F *hist1 = (TH1F*)file1->Get("dimuon_mass"); //benim verim
  TH1F *hist2 = (TH1F*)file2->Get("Dimuon_mass"); //tut verisi 

  int nBins1 = hist1->GetNbinsX(); //Int_t  RooPlot::GetNbinsX() const { return _hist->GetNbinsX() ; } 
  int nBins2 = hist2->GetNbinsX();

  bool Equal; 

if (nBins1 == nBins2) {
    Equal = true; // Eğer histogramlar farklı sayıda bine sahipse, eşitlik sağlanamaz.
    std::cout << "ife girdi" << endl;
} /*else { 
  for (int i = 1; i <= nBins1; ++i) { //binleri karşılaştır, != histogramları getbincontentle karşılaştır 
    if (hist1->GetBinContent(i) == hist2->GetBinContent(i)){
      Equal = false;
      break;
    std::cout << "else'e girdi girdi" << endl;

  }} */ //burası Y binlerinin de dahil olabilmesi için ama 2 boyutlu bir histogramımın mı olması gerekir? 

  if (Equal) {
    std::cout << "Histograms are identical." << std::endl;
  } else {
    std::cout << "Histograms are different." << std::endl;
  }


delete hist1;
delete hist2;
file1->Close();
file2->Close();
delete file1;
delete file2;


}

# Behavioral-Cloning
Build a CNN model, Train it, then run the training model in a car simulate  
---




## Project Brief 
---


[//]: # (Image References)


[Histogram_Orig_data]: ./figures/Histogram_Original_dataset.png "Original Data Histogram"
[Histogram_Proc_data]: ./figures/Histogram_Processed_dataset.png "Processed Data Histogram"
[original_cropp_image]: ./figures/original_cropp_image.png "original_cropp_image"
[model_loss_graph]: ./figures/model_loss_graph.png "model_loss_graph"
[original_cropp_image]: ./figures/original_cropp_image.png "original_cropp_image"




## Dataset
For this project, we were supposed to generate our own data running the Udacity Car Simulator.  However, we also were given a set of data in case we needed.  I used the given data set as I was not able to generate a large enough set to train the model. 

### Exploration and preparation of the Data
From the given data set, I made a copy of the original CSV log file and randomly deleted about 70% of the registered images with steering angle of zero.  The purpose was to smooth the original dataset distribution show Original Data Histogram below.  In effort to augment the quantity of images, I create new images by flipping or mirroring those images with steering angle value different to zero, and adjusted the steering angle for these images accordantly.  A new dataset distribution can be appreciated in Processed Data Histogram below.  Twenty percent of this data was randomly selected a side for testing purpose.
![Histogram_Orig_data]
![Histogram_Proc_data]

### Preparing the data
To improve the outcome of the training, the model normalizes and crop the images, as illustrate in the figure below.  

![original_cropp_image]

---
## The model
I started this project using a LetNet5 model build in Keras.  After several changes to the LeNet5 model and unable to accomplish the objective, I shift to the proposed [“end-to-end” model used by NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).  Still, I added two Dropout layers no included in the original NVIDIA model and modify the number of hidden layers, the new model shown in the table below.  This new model was run using Adam optimization for 4 epochs with batch size set to 32 and Generator that fed the data to it in a more memory-efficient way.




<html xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:w="urn:schemas-microsoft-com:office:word"
xmlns:m="http://schemas.microsoft.com/office/2004/12/omml"
xmlns="http://www.w3.org/TR/REC-html40">

<head>
<meta http-equiv=Content-Type content="text/html; charset=windows-1252">
<meta name=ProgId content=Word.Document>
<meta name=Generator content="Microsoft Word 15">
<meta name=Originator content="Microsoft Word 15">
<link rel=File-List href="layes_table_files/filelist.xml">
<!--[if gte mso 9]><xml>
 <o:DocumentProperties>
  <o:Author>Luis Riera</o:Author>
  <o:LastAuthor>Luis Riera</o:LastAuthor>
  <o:Revision>1</o:Revision>
  <o:TotalTime>2</o:TotalTime>
  <o:Created>2017-07-15T23:04:00Z</o:Created>
  <o:LastSaved>2017-07-15T23:06:00Z</o:LastSaved>
  <o:Pages>1</o:Pages>
  <o:Words>138</o:Words>
  <o:Characters>790</o:Characters>
  <o:Lines>6</o:Lines>
  <o:Paragraphs>1</o:Paragraphs>
  <o:CharactersWithSpaces>927</o:CharactersWithSpaces>
  <o:Version>16.00</o:Version>
 </o:DocumentProperties>
 <o:OfficeDocumentSettings>
  <o:AllowPNG/>
 </o:OfficeDocumentSettings>
</xml><![endif]-->
<link rel=themeData href="layes_table_files/themedata.thmx">
<link rel=colorSchemeMapping href="layes_table_files/colorschememapping.xml">
<!--[if gte mso 9]><xml>
 <w:WordDocument>
  <w:SpellingState>Clean</w:SpellingState>
  <w:GrammarState>Clean</w:GrammarState>
  <w:TrackMoves>false</w:TrackMoves>
  <w:TrackFormatting/>
  <w:HyphenationZone>21</w:HyphenationZone>
  <w:PunctuationKerning/>
  <w:ValidateAgainstSchemas/>
  <w:SaveIfXMLInvalid>false</w:SaveIfXMLInvalid>
  <w:IgnoreMixedContent>false</w:IgnoreMixedContent>
  <w:AlwaysShowPlaceholderText>false</w:AlwaysShowPlaceholderText>
  <w:DoNotPromoteQF/>
  <w:LidThemeOther>ES-VE</w:LidThemeOther>
  <w:LidThemeAsian>X-NONE</w:LidThemeAsian>
  <w:LidThemeComplexScript>X-NONE</w:LidThemeComplexScript>
  <w:Compatibility>
   <w:BreakWrappedTables/>
   <w:SnapToGridInCell/>
   <w:WrapTextWithPunct/>
   <w:UseAsianBreakRules/>
   <w:UseWord2010TableStyleRules/>
   <w:DontGrowAutofit/>
   <w:SplitPgBreakAndParaMark/>
   <w:EnableOpenTypeKerning/>
   <w:DontFlipMirrorIndents/>
   <w:OverrideTableStyleHps/>
  </w:Compatibility>
  <m:mathPr>
   <m:mathFont m:val="Cambria Math"/>
   <m:brkBin m:val="before"/>
   <m:brkBinSub m:val="&#45;-"/>
   <m:smallFrac m:val="off"/>
   <m:dispDef/>
   <m:lMargin m:val="0"/>
   <m:rMargin m:val="0"/>
   <m:defJc m:val="centerGroup"/>
   <m:wrapIndent m:val="1440"/>
   <m:intLim m:val="subSup"/>
   <m:naryLim m:val="undOvr"/>
  </m:mathPr></w:WordDocument>
</xml><![endif]--><!--[if gte mso 9]><xml>
 <w:LatentStyles DefLockedState="false" DefUnhideWhenUsed="false"
  DefSemiHidden="false" DefQFormat="false" DefPriority="99"
  LatentStyleCount="375">
  <w:LsdException Locked="false" Priority="0" QFormat="true" Name="Normal"/>
  <w:LsdException Locked="false" Priority="9" QFormat="true" Name="heading 1"/>
  <w:LsdException Locked="false" Priority="9" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="heading 2"/>
  <w:LsdException Locked="false" Priority="9" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="heading 3"/>
  <w:LsdException Locked="false" Priority="9" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="heading 4"/>
  <w:LsdException Locked="false" Priority="9" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="heading 5"/>
  <w:LsdException Locked="false" Priority="9" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="heading 6"/>
  <w:LsdException Locked="false" Priority="9" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="heading 7"/>
  <w:LsdException Locked="false" Priority="9" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="heading 8"/>
  <w:LsdException Locked="false" Priority="9" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="heading 9"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 5"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 6"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 7"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 8"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index 9"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 1"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 2"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 3"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 4"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 5"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 6"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 7"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 8"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" Name="toc 9"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Normal Indent"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="footnote text"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="annotation text"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="header"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="footer"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="index heading"/>
  <w:LsdException Locked="false" Priority="35" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="caption"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="table of figures"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="envelope address"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="envelope return"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="footnote reference"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="annotation reference"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="line number"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="page number"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="endnote reference"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="endnote text"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="table of authorities"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="macro"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="toa heading"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Bullet"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Number"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List 5"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Bullet 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Bullet 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Bullet 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Bullet 5"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Number 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Number 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Number 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Number 5"/>
  <w:LsdException Locked="false" Priority="10" QFormat="true" Name="Title"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Closing"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Signature"/>
  <w:LsdException Locked="false" Priority="1" SemiHidden="true"
   UnhideWhenUsed="true" Name="Default Paragraph Font"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Body Text"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Body Text Indent"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Continue"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Continue 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Continue 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Continue 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="List Continue 5"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Message Header"/>
  <w:LsdException Locked="false" Priority="11" QFormat="true" Name="Subtitle"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Salutation"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Date"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Body Text First Indent"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Body Text First Indent 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Note Heading"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Body Text 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Body Text 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Body Text Indent 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Body Text Indent 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Block Text"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Hyperlink"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="FollowedHyperlink"/>
  <w:LsdException Locked="false" Priority="22" QFormat="true" Name="Strong"/>
  <w:LsdException Locked="false" Priority="20" QFormat="true" Name="Emphasis"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Document Map"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Plain Text"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="E-mail Signature"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Top of Form"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Bottom of Form"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Normal (Web)"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Acronym"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Address"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Cite"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Code"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Definition"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Keyboard"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Preformatted"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Sample"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Typewriter"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="HTML Variable"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Normal Table"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="annotation subject"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="No List"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Outline List 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Outline List 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Outline List 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Simple 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Simple 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Simple 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Classic 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Classic 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Classic 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Classic 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Colorful 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Colorful 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Colorful 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Columns 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Columns 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Columns 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Columns 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Columns 5"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Grid 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Grid 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Grid 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Grid 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Grid 5"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Grid 6"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Grid 7"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Grid 8"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table List 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table List 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table List 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table List 4"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table List 5"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table List 6"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table List 7"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table List 8"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table 3D effects 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table 3D effects 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table 3D effects 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Contemporary"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Elegant"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Professional"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Subtle 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Subtle 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Web 1"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Web 2"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Web 3"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Balloon Text"/>
  <w:LsdException Locked="false" Priority="59" Name="Table Grid"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Table Theme"/>
  <w:LsdException Locked="false" SemiHidden="true" Name="Placeholder Text"/>
  <w:LsdException Locked="false" Priority="1" QFormat="true" Name="No Spacing"/>
  <w:LsdException Locked="false" Priority="60" Name="Light Shading"/>
  <w:LsdException Locked="false" Priority="61" Name="Light List"/>
  <w:LsdException Locked="false" Priority="62" Name="Light Grid"/>
  <w:LsdException Locked="false" Priority="63" Name="Medium Shading 1"/>
  <w:LsdException Locked="false" Priority="64" Name="Medium Shading 2"/>
  <w:LsdException Locked="false" Priority="65" Name="Medium List 1"/>
  <w:LsdException Locked="false" Priority="66" Name="Medium List 2"/>
  <w:LsdException Locked="false" Priority="67" Name="Medium Grid 1"/>
  <w:LsdException Locked="false" Priority="68" Name="Medium Grid 2"/>
  <w:LsdException Locked="false" Priority="69" Name="Medium Grid 3"/>
  <w:LsdException Locked="false" Priority="70" Name="Dark List"/>
  <w:LsdException Locked="false" Priority="71" Name="Colorful Shading"/>
  <w:LsdException Locked="false" Priority="72" Name="Colorful List"/>
  <w:LsdException Locked="false" Priority="73" Name="Colorful Grid"/>
  <w:LsdException Locked="false" Priority="60" Name="Light Shading Accent 1"/>
  <w:LsdException Locked="false" Priority="61" Name="Light List Accent 1"/>
  <w:LsdException Locked="false" Priority="62" Name="Light Grid Accent 1"/>
  <w:LsdException Locked="false" Priority="63" Name="Medium Shading 1 Accent 1"/>
  <w:LsdException Locked="false" Priority="64" Name="Medium Shading 2 Accent 1"/>
  <w:LsdException Locked="false" Priority="65" Name="Medium List 1 Accent 1"/>
  <w:LsdException Locked="false" SemiHidden="true" Name="Revision"/>
  <w:LsdException Locked="false" Priority="34" QFormat="true"
   Name="List Paragraph"/>
  <w:LsdException Locked="false" Priority="29" QFormat="true" Name="Quote"/>
  <w:LsdException Locked="false" Priority="30" QFormat="true"
   Name="Intense Quote"/>
  <w:LsdException Locked="false" Priority="66" Name="Medium List 2 Accent 1"/>
  <w:LsdException Locked="false" Priority="67" Name="Medium Grid 1 Accent 1"/>
  <w:LsdException Locked="false" Priority="68" Name="Medium Grid 2 Accent 1"/>
  <w:LsdException Locked="false" Priority="69" Name="Medium Grid 3 Accent 1"/>
  <w:LsdException Locked="false" Priority="70" Name="Dark List Accent 1"/>
  <w:LsdException Locked="false" Priority="71" Name="Colorful Shading Accent 1"/>
  <w:LsdException Locked="false" Priority="72" Name="Colorful List Accent 1"/>
  <w:LsdException Locked="false" Priority="73" Name="Colorful Grid Accent 1"/>
  <w:LsdException Locked="false" Priority="60" Name="Light Shading Accent 2"/>
  <w:LsdException Locked="false" Priority="61" Name="Light List Accent 2"/>
  <w:LsdException Locked="false" Priority="62" Name="Light Grid Accent 2"/>
  <w:LsdException Locked="false" Priority="63" Name="Medium Shading 1 Accent 2"/>
  <w:LsdException Locked="false" Priority="64" Name="Medium Shading 2 Accent 2"/>
  <w:LsdException Locked="false" Priority="65" Name="Medium List 1 Accent 2"/>
  <w:LsdException Locked="false" Priority="66" Name="Medium List 2 Accent 2"/>
  <w:LsdException Locked="false" Priority="67" Name="Medium Grid 1 Accent 2"/>
  <w:LsdException Locked="false" Priority="68" Name="Medium Grid 2 Accent 2"/>
  <w:LsdException Locked="false" Priority="69" Name="Medium Grid 3 Accent 2"/>
  <w:LsdException Locked="false" Priority="70" Name="Dark List Accent 2"/>
  <w:LsdException Locked="false" Priority="71" Name="Colorful Shading Accent 2"/>
  <w:LsdException Locked="false" Priority="72" Name="Colorful List Accent 2"/>
  <w:LsdException Locked="false" Priority="73" Name="Colorful Grid Accent 2"/>
  <w:LsdException Locked="false" Priority="60" Name="Light Shading Accent 3"/>
  <w:LsdException Locked="false" Priority="61" Name="Light List Accent 3"/>
  <w:LsdException Locked="false" Priority="62" Name="Light Grid Accent 3"/>
  <w:LsdException Locked="false" Priority="63" Name="Medium Shading 1 Accent 3"/>
  <w:LsdException Locked="false" Priority="64" Name="Medium Shading 2 Accent 3"/>
  <w:LsdException Locked="false" Priority="65" Name="Medium List 1 Accent 3"/>
  <w:LsdException Locked="false" Priority="66" Name="Medium List 2 Accent 3"/>
  <w:LsdException Locked="false" Priority="67" Name="Medium Grid 1 Accent 3"/>
  <w:LsdException Locked="false" Priority="68" Name="Medium Grid 2 Accent 3"/>
  <w:LsdException Locked="false" Priority="69" Name="Medium Grid 3 Accent 3"/>
  <w:LsdException Locked="false" Priority="70" Name="Dark List Accent 3"/>
  <w:LsdException Locked="false" Priority="71" Name="Colorful Shading Accent 3"/>
  <w:LsdException Locked="false" Priority="72" Name="Colorful List Accent 3"/>
  <w:LsdException Locked="false" Priority="73" Name="Colorful Grid Accent 3"/>
  <w:LsdException Locked="false" Priority="60" Name="Light Shading Accent 4"/>
  <w:LsdException Locked="false" Priority="61" Name="Light List Accent 4"/>
  <w:LsdException Locked="false" Priority="62" Name="Light Grid Accent 4"/>
  <w:LsdException Locked="false" Priority="63" Name="Medium Shading 1 Accent 4"/>
  <w:LsdException Locked="false" Priority="64" Name="Medium Shading 2 Accent 4"/>
  <w:LsdException Locked="false" Priority="65" Name="Medium List 1 Accent 4"/>
  <w:LsdException Locked="false" Priority="66" Name="Medium List 2 Accent 4"/>
  <w:LsdException Locked="false" Priority="67" Name="Medium Grid 1 Accent 4"/>
  <w:LsdException Locked="false" Priority="68" Name="Medium Grid 2 Accent 4"/>
  <w:LsdException Locked="false" Priority="69" Name="Medium Grid 3 Accent 4"/>
  <w:LsdException Locked="false" Priority="70" Name="Dark List Accent 4"/>
  <w:LsdException Locked="false" Priority="71" Name="Colorful Shading Accent 4"/>
  <w:LsdException Locked="false" Priority="72" Name="Colorful List Accent 4"/>
  <w:LsdException Locked="false" Priority="73" Name="Colorful Grid Accent 4"/>
  <w:LsdException Locked="false" Priority="60" Name="Light Shading Accent 5"/>
  <w:LsdException Locked="false" Priority="61" Name="Light List Accent 5"/>
  <w:LsdException Locked="false" Priority="62" Name="Light Grid Accent 5"/>
  <w:LsdException Locked="false" Priority="63" Name="Medium Shading 1 Accent 5"/>
  <w:LsdException Locked="false" Priority="64" Name="Medium Shading 2 Accent 5"/>
  <w:LsdException Locked="false" Priority="65" Name="Medium List 1 Accent 5"/>
  <w:LsdException Locked="false" Priority="66" Name="Medium List 2 Accent 5"/>
  <w:LsdException Locked="false" Priority="67" Name="Medium Grid 1 Accent 5"/>
  <w:LsdException Locked="false" Priority="68" Name="Medium Grid 2 Accent 5"/>
  <w:LsdException Locked="false" Priority="69" Name="Medium Grid 3 Accent 5"/>
  <w:LsdException Locked="false" Priority="70" Name="Dark List Accent 5"/>
  <w:LsdException Locked="false" Priority="71" Name="Colorful Shading Accent 5"/>
  <w:LsdException Locked="false" Priority="72" Name="Colorful List Accent 5"/>
  <w:LsdException Locked="false" Priority="73" Name="Colorful Grid Accent 5"/>
  <w:LsdException Locked="false" Priority="60" Name="Light Shading Accent 6"/>
  <w:LsdException Locked="false" Priority="61" Name="Light List Accent 6"/>
  <w:LsdException Locked="false" Priority="62" Name="Light Grid Accent 6"/>
  <w:LsdException Locked="false" Priority="63" Name="Medium Shading 1 Accent 6"/>
  <w:LsdException Locked="false" Priority="64" Name="Medium Shading 2 Accent 6"/>
  <w:LsdException Locked="false" Priority="65" Name="Medium List 1 Accent 6"/>
  <w:LsdException Locked="false" Priority="66" Name="Medium List 2 Accent 6"/>
  <w:LsdException Locked="false" Priority="67" Name="Medium Grid 1 Accent 6"/>
  <w:LsdException Locked="false" Priority="68" Name="Medium Grid 2 Accent 6"/>
  <w:LsdException Locked="false" Priority="69" Name="Medium Grid 3 Accent 6"/>
  <w:LsdException Locked="false" Priority="70" Name="Dark List Accent 6"/>
  <w:LsdException Locked="false" Priority="71" Name="Colorful Shading Accent 6"/>
  <w:LsdException Locked="false" Priority="72" Name="Colorful List Accent 6"/>
  <w:LsdException Locked="false" Priority="73" Name="Colorful Grid Accent 6"/>
  <w:LsdException Locked="false" Priority="19" QFormat="true"
   Name="Subtle Emphasis"/>
  <w:LsdException Locked="false" Priority="21" QFormat="true"
   Name="Intense Emphasis"/>
  <w:LsdException Locked="false" Priority="31" QFormat="true"
   Name="Subtle Reference"/>
  <w:LsdException Locked="false" Priority="32" QFormat="true"
   Name="Intense Reference"/>
  <w:LsdException Locked="false" Priority="33" QFormat="true" Name="Book Title"/>
  <w:LsdException Locked="false" Priority="37" SemiHidden="true"
   UnhideWhenUsed="true" Name="Bibliography"/>
  <w:LsdException Locked="false" Priority="39" SemiHidden="true"
   UnhideWhenUsed="true" QFormat="true" Name="TOC Heading"/>
  <w:LsdException Locked="false" Priority="41" Name="Plain Table 1"/>
  <w:LsdException Locked="false" Priority="42" Name="Plain Table 2"/>
  <w:LsdException Locked="false" Priority="43" Name="Plain Table 3"/>
  <w:LsdException Locked="false" Priority="44" Name="Plain Table 4"/>
  <w:LsdException Locked="false" Priority="45" Name="Plain Table 5"/>
  <w:LsdException Locked="false" Priority="40" Name="Grid Table Light"/>
  <w:LsdException Locked="false" Priority="46" Name="Grid Table 1 Light"/>
  <w:LsdException Locked="false" Priority="47" Name="Grid Table 2"/>
  <w:LsdException Locked="false" Priority="48" Name="Grid Table 3"/>
  <w:LsdException Locked="false" Priority="49" Name="Grid Table 4"/>
  <w:LsdException Locked="false" Priority="50" Name="Grid Table 5 Dark"/>
  <w:LsdException Locked="false" Priority="51" Name="Grid Table 6 Colorful"/>
  <w:LsdException Locked="false" Priority="52" Name="Grid Table 7 Colorful"/>
  <w:LsdException Locked="false" Priority="46"
   Name="Grid Table 1 Light Accent 1"/>
  <w:LsdException Locked="false" Priority="47" Name="Grid Table 2 Accent 1"/>
  <w:LsdException Locked="false" Priority="48" Name="Grid Table 3 Accent 1"/>
  <w:LsdException Locked="false" Priority="49" Name="Grid Table 4 Accent 1"/>
  <w:LsdException Locked="false" Priority="50" Name="Grid Table 5 Dark Accent 1"/>
  <w:LsdException Locked="false" Priority="51"
   Name="Grid Table 6 Colorful Accent 1"/>
  <w:LsdException Locked="false" Priority="52"
   Name="Grid Table 7 Colorful Accent 1"/>
  <w:LsdException Locked="false" Priority="46"
   Name="Grid Table 1 Light Accent 2"/>
  <w:LsdException Locked="false" Priority="47" Name="Grid Table 2 Accent 2"/>
  <w:LsdException Locked="false" Priority="48" Name="Grid Table 3 Accent 2"/>
  <w:LsdException Locked="false" Priority="49" Name="Grid Table 4 Accent 2"/>
  <w:LsdException Locked="false" Priority="50" Name="Grid Table 5 Dark Accent 2"/>
  <w:LsdException Locked="false" Priority="51"
   Name="Grid Table 6 Colorful Accent 2"/>
  <w:LsdException Locked="false" Priority="52"
   Name="Grid Table 7 Colorful Accent 2"/>
  <w:LsdException Locked="false" Priority="46"
   Name="Grid Table 1 Light Accent 3"/>
  <w:LsdException Locked="false" Priority="47" Name="Grid Table 2 Accent 3"/>
  <w:LsdException Locked="false" Priority="48" Name="Grid Table 3 Accent 3"/>
  <w:LsdException Locked="false" Priority="49" Name="Grid Table 4 Accent 3"/>
  <w:LsdException Locked="false" Priority="50" Name="Grid Table 5 Dark Accent 3"/>
  <w:LsdException Locked="false" Priority="51"
   Name="Grid Table 6 Colorful Accent 3"/>
  <w:LsdException Locked="false" Priority="52"
   Name="Grid Table 7 Colorful Accent 3"/>
  <w:LsdException Locked="false" Priority="46"
   Name="Grid Table 1 Light Accent 4"/>
  <w:LsdException Locked="false" Priority="47" Name="Grid Table 2 Accent 4"/>
  <w:LsdException Locked="false" Priority="48" Name="Grid Table 3 Accent 4"/>
  <w:LsdException Locked="false" Priority="49" Name="Grid Table 4 Accent 4"/>
  <w:LsdException Locked="false" Priority="50" Name="Grid Table 5 Dark Accent 4"/>
  <w:LsdException Locked="false" Priority="51"
   Name="Grid Table 6 Colorful Accent 4"/>
  <w:LsdException Locked="false" Priority="52"
   Name="Grid Table 7 Colorful Accent 4"/>
  <w:LsdException Locked="false" Priority="46"
   Name="Grid Table 1 Light Accent 5"/>
  <w:LsdException Locked="false" Priority="47" Name="Grid Table 2 Accent 5"/>
  <w:LsdException Locked="false" Priority="48" Name="Grid Table 3 Accent 5"/>
  <w:LsdException Locked="false" Priority="49" Name="Grid Table 4 Accent 5"/>
  <w:LsdException Locked="false" Priority="50" Name="Grid Table 5 Dark Accent 5"/>
  <w:LsdException Locked="false" Priority="51"
   Name="Grid Table 6 Colorful Accent 5"/>
  <w:LsdException Locked="false" Priority="52"
   Name="Grid Table 7 Colorful Accent 5"/>
  <w:LsdException Locked="false" Priority="46"
   Name="Grid Table 1 Light Accent 6"/>
  <w:LsdException Locked="false" Priority="47" Name="Grid Table 2 Accent 6"/>
  <w:LsdException Locked="false" Priority="48" Name="Grid Table 3 Accent 6"/>
  <w:LsdException Locked="false" Priority="49" Name="Grid Table 4 Accent 6"/>
  <w:LsdException Locked="false" Priority="50" Name="Grid Table 5 Dark Accent 6"/>
  <w:LsdException Locked="false" Priority="51"
   Name="Grid Table 6 Colorful Accent 6"/>
  <w:LsdException Locked="false" Priority="52"
   Name="Grid Table 7 Colorful Accent 6"/>
  <w:LsdException Locked="false" Priority="46" Name="List Table 1 Light"/>
  <w:LsdException Locked="false" Priority="47" Name="List Table 2"/>
  <w:LsdException Locked="false" Priority="48" Name="List Table 3"/>
  <w:LsdException Locked="false" Priority="49" Name="List Table 4"/>
  <w:LsdException Locked="false" Priority="50" Name="List Table 5 Dark"/>
  <w:LsdException Locked="false" Priority="51" Name="List Table 6 Colorful"/>
  <w:LsdException Locked="false" Priority="52" Name="List Table 7 Colorful"/>
  <w:LsdException Locked="false" Priority="46"
   Name="List Table 1 Light Accent 1"/>
  <w:LsdException Locked="false" Priority="47" Name="List Table 2 Accent 1"/>
  <w:LsdException Locked="false" Priority="48" Name="List Table 3 Accent 1"/>
  <w:LsdException Locked="false" Priority="49" Name="List Table 4 Accent 1"/>
  <w:LsdException Locked="false" Priority="50" Name="List Table 5 Dark Accent 1"/>
  <w:LsdException Locked="false" Priority="51"
   Name="List Table 6 Colorful Accent 1"/>
  <w:LsdException Locked="false" Priority="52"
   Name="List Table 7 Colorful Accent 1"/>
  <w:LsdException Locked="false" Priority="46"
   Name="List Table 1 Light Accent 2"/>
  <w:LsdException Locked="false" Priority="47" Name="List Table 2 Accent 2"/>
  <w:LsdException Locked="false" Priority="48" Name="List Table 3 Accent 2"/>
  <w:LsdException Locked="false" Priority="49" Name="List Table 4 Accent 2"/>
  <w:LsdException Locked="false" Priority="50" Name="List Table 5 Dark Accent 2"/>
  <w:LsdException Locked="false" Priority="51"
   Name="List Table 6 Colorful Accent 2"/>
  <w:LsdException Locked="false" Priority="52"
   Name="List Table 7 Colorful Accent 2"/>
  <w:LsdException Locked="false" Priority="46"
   Name="List Table 1 Light Accent 3"/>
  <w:LsdException Locked="false" Priority="47" Name="List Table 2 Accent 3"/>
  <w:LsdException Locked="false" Priority="48" Name="List Table 3 Accent 3"/>
  <w:LsdException Locked="false" Priority="49" Name="List Table 4 Accent 3"/>
  <w:LsdException Locked="false" Priority="50" Name="List Table 5 Dark Accent 3"/>
  <w:LsdException Locked="false" Priority="51"
   Name="List Table 6 Colorful Accent 3"/>
  <w:LsdException Locked="false" Priority="52"
   Name="List Table 7 Colorful Accent 3"/>
  <w:LsdException Locked="false" Priority="46"
   Name="List Table 1 Light Accent 4"/>
  <w:LsdException Locked="false" Priority="47" Name="List Table 2 Accent 4"/>
  <w:LsdException Locked="false" Priority="48" Name="List Table 3 Accent 4"/>
  <w:LsdException Locked="false" Priority="49" Name="List Table 4 Accent 4"/>
  <w:LsdException Locked="false" Priority="50" Name="List Table 5 Dark Accent 4"/>
  <w:LsdException Locked="false" Priority="51"
   Name="List Table 6 Colorful Accent 4"/>
  <w:LsdException Locked="false" Priority="52"
   Name="List Table 7 Colorful Accent 4"/>
  <w:LsdException Locked="false" Priority="46"
   Name="List Table 1 Light Accent 5"/>
  <w:LsdException Locked="false" Priority="47" Name="List Table 2 Accent 5"/>
  <w:LsdException Locked="false" Priority="48" Name="List Table 3 Accent 5"/>
  <w:LsdException Locked="false" Priority="49" Name="List Table 4 Accent 5"/>
  <w:LsdException Locked="false" Priority="50" Name="List Table 5 Dark Accent 5"/>
  <w:LsdException Locked="false" Priority="51"
   Name="List Table 6 Colorful Accent 5"/>
  <w:LsdException Locked="false" Priority="52"
   Name="List Table 7 Colorful Accent 5"/>
  <w:LsdException Locked="false" Priority="46"
   Name="List Table 1 Light Accent 6"/>
  <w:LsdException Locked="false" Priority="47" Name="List Table 2 Accent 6"/>
  <w:LsdException Locked="false" Priority="48" Name="List Table 3 Accent 6"/>
  <w:LsdException Locked="false" Priority="49" Name="List Table 4 Accent 6"/>
  <w:LsdException Locked="false" Priority="50" Name="List Table 5 Dark Accent 6"/>
  <w:LsdException Locked="false" Priority="51"
   Name="List Table 6 Colorful Accent 6"/>
  <w:LsdException Locked="false" Priority="52"
   Name="List Table 7 Colorful Accent 6"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Mention"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Smart Hyperlink"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Hashtag"/>
  <w:LsdException Locked="false" SemiHidden="true" UnhideWhenUsed="true"
   Name="Unresolved Mention"/>
 </w:LatentStyles>
</xml><![endif]-->
<style>
<!--
 /* Font Definitions */
 @font-face
	{font-family:"Cambria Math";
	panose-1:2 4 5 3 5 4 6 3 2 4;
	mso-font-charset:1;
	mso-generic-font-family:roman;
	mso-font-pitch:variable;
	mso-font-signature:0 0 0 0 0 0;}
@font-face
	{font-family:Calibri;
	panose-1:2 15 5 2 2 2 4 3 2 4;
	mso-font-charset:0;
	mso-generic-font-family:swiss;
	mso-font-pitch:variable;
	mso-font-signature:-536859905 -1073732485 9 0 511 0;}
 /* Style Definitions */
 p.MsoNormal, li.MsoNormal, div.MsoNormal
	{mso-style-unhide:no;
	mso-style-qformat:yes;
	mso-style-parent:"";
	margin-top:0in;
	margin-right:0in;
	margin-bottom:10.0pt;
	margin-left:0in;
	line-height:115%;
	mso-pagination:widow-orphan;
	font-size:11.0pt;
	font-family:"Calibri",sans-serif;
	mso-ascii-font-family:Calibri;
	mso-ascii-theme-font:minor-latin;
	mso-fareast-font-family:Calibri;
	mso-fareast-theme-font:minor-latin;
	mso-hansi-font-family:Calibri;
	mso-hansi-theme-font:minor-latin;
	mso-bidi-font-family:"Times New Roman";
	mso-bidi-theme-font:minor-bidi;
	mso-ansi-language:ES-VE;}
.MsoChpDefault
	{mso-style-type:export-only;
	mso-default-props:yes;
	font-family:"Calibri",sans-serif;
	mso-ascii-font-family:Calibri;
	mso-ascii-theme-font:minor-latin;
	mso-fareast-font-family:Calibri;
	mso-fareast-theme-font:minor-latin;
	mso-hansi-font-family:Calibri;
	mso-hansi-theme-font:minor-latin;
	mso-bidi-font-family:"Times New Roman";
	mso-bidi-theme-font:minor-bidi;
	mso-ansi-language:ES-VE;}
.MsoPapDefault
	{mso-style-type:export-only;
	margin-bottom:10.0pt;
	line-height:115%;}
@page WordSection1
	{size:8.5in 11.0in;
	margin:1.0in 1.0in 1.0in 1.0in;
	mso-header-margin:.5in;
	mso-footer-margin:.5in;
	mso-paper-source:0;}
div.WordSection1
	{page:WordSection1;}
-->
</style>
<!--[if gte mso 10]>
<style>
 /* Style Definitions */
 table.MsoNormalTable
	{mso-style-name:"Table Normal";
	mso-tstyle-rowband-size:0;
	mso-tstyle-colband-size:0;
	mso-style-noshow:yes;
	mso-style-priority:99;
	mso-style-parent:"";
	mso-padding-alt:0in 5.4pt 0in 5.4pt;
	mso-para-margin-top:0in;
	mso-para-margin-right:0in;
	mso-para-margin-bottom:10.0pt;
	mso-para-margin-left:0in;
	line-height:115%;
	mso-pagination:widow-orphan;
	font-size:11.0pt;
	font-family:"Calibri",sans-serif;
	mso-ascii-font-family:Calibri;
	mso-ascii-theme-font:minor-latin;
	mso-hansi-font-family:Calibri;
	mso-hansi-theme-font:minor-latin;
	mso-bidi-font-family:"Times New Roman";
	mso-bidi-theme-font:minor-bidi;
	mso-ansi-language:ES-VE;}
</style>
<![endif]--><!--[if gte mso 9]><xml>
 <o:shapedefaults v:ext="edit" spidmax="1026"/>
</xml><![endif]--><!--[if gte mso 9]><xml>
 <o:shapelayout v:ext="edit">
  <o:idmap v:ext="edit" data="1"/>
 </o:shapelayout></xml><![endif]-->
</head>

<body lang=EN-US style='tab-interval:35.4pt'>

<div class=WordSection1>

<table class=MsoNormalTable border=0 cellspacing=0 cellpadding=0 width=672
 style='width:7.0in;margin-left:5.65pt;border-collapse:collapse;mso-yfti-tbllook:
 1184;mso-padding-left-alt:5.4pt;mso-padding-right-alt:5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border:solid #8EA9DB 1.0pt;
  border-right:none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-left-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;background:#4472C4;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><b><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:white;mso-ansi-language:EN-US'>Layer (type)<o:p></o:p></span></b></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border-top:solid #8EA9DB 1.0pt;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:none;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  background:#4472C4;padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><b><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:white;mso-ansi-language:EN-US'>Output Shape<o:p></o:p></span></b></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border-top:solid #8EA9DB 1.0pt;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:none;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  background:#4472C4;padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><b><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:white;mso-ansi-language:EN-US'>Param #<o:p></o:p></span></b></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border:solid #8EA9DB 1.0pt;
  border-left:none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:
  solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;background:#4472C4;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><b><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:white;mso-ansi-language:EN-US'>Connected to<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>cropping2d_1 (Cropping2D)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 65, 318, 3)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>0<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>cropping2d_input_1[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>lambda_1 (Lambda)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 65, 318, 3)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>0<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>cropping2d_1[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_1 (Convolution2D)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 31, 157, 24)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>1824<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>lambda_1[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_2 (Convolution2D)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 14, 77, 36)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>21636<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_1[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_3 (Convolution2D)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 5, 37, 48)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>43248<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_2[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_4 (Convolution2D)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 3, 35, 64)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>27712<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_3[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_5 (Convolution2D)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 1, 33, 64)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>36928<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_4[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>flatten_1 (Flatten)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 2112)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>0<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>convolution2d_5[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dense_1 (Dense)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 1300)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>2746900<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>flatten_1[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dropout_1 (Dropout)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 1300)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>0<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dense_1[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dense_2 (Dense)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 160)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>208160<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dropout_1[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dropout_2 (Dropout)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 160)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>0<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dense_2[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dense_3 (Dense)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 50)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>8050<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dropout_2[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:14;mso-yfti-lastrow:yes;height:.2in'>
  <td width=252 nowrap valign=bottom style='width:188.75pt;border-top:none;
  border-left:solid #8EA9DB 1.0pt;border-bottom:solid #8EA9DB 1.0pt;border-right:
  none;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-left-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dense_4 (Dense)<o:p></o:p></span></p>
  </td>
  <td width=156 nowrap valign=bottom style='width:117.0pt;border:none;
  border-bottom:solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>(None, 1)<o:p></o:p></span></p>
  </td>
  <td width=96 nowrap valign=bottom style='width:1.0in;border:none;border-bottom:
  solid #8EA9DB 1.0pt;mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:
  solid #8EA9DB .5pt;mso-border-bottom-alt:solid #8EA9DB .5pt;padding:.75pt 5.4pt .75pt 5.4pt;
  height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>51<o:p></o:p></span></p>
  </td>
  <td width=168 nowrap valign=bottom style='width:126.25pt;border-top:none;
  border-left:none;border-bottom:solid #8EA9DB 1.0pt;border-right:solid #8EA9DB 1.0pt;
  mso-border-top-alt:solid #8EA9DB .5pt;mso-border-top-alt:solid #8EA9DB .5pt;
  mso-border-bottom-alt:solid #8EA9DB .5pt;mso-border-right-alt:solid #8EA9DB .5pt;
  padding:.75pt 5.4pt .75pt 5.4pt;height:.2in'>
  <p class=MsoNormal style='margin-bottom:0in;margin-bottom:.0001pt;line-height:
  normal'><span style='mso-ascii-font-family:Calibri;mso-fareast-font-family:
  "Times New Roman";mso-hansi-font-family:Calibri;mso-bidi-font-family:Calibri;
  color:black;mso-ansi-language:EN-US'>dense_3[0][0]<o:p></o:p></span></p>
  </td>
 </tr>
</table>

<p class=MsoNormal><span style='mso-spacerun:yes'> </span></p>

</div>

</body>

</html>

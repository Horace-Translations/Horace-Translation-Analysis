import string
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

translations = [
 { 
        "text": "What pretty youth, weltering in roses With liquid odors overspread, O Pirrha thee in his arms incloses, When thou loves Lecture has him read in the inner bower: Neglecting curious dresses, For whom plaitest thou the gold wire of thy tresses? How often will he that at his pleasure Enjoys thee now (alas) complain, That he is robbed of that sweet treasure By angry gods, and vows made vain? How will he curse the Seas so soon that wrangle, Whom such sly baits could not before intangle? For he poor soul, deceived, believed Thou wouldst be true to him alone, And lovely: But his heart, now grieved, Thy false inconstancy doth mone. His tents he in destructions black field pitches, Whom thou untride, with thy fair face bewitches. The Temples wall, thats consecrated, To every eye the Table shows Where my sad ship-wreck is related: And how in the midst of all my woes, I hung to the Sea god, after strange beseeches, My doublet wringing wet, and cod-piect breeches",
        "author": "John Ashmore",
        "date": 1621
    },
    {
        "text": "What young man, Pyrrha, doth thee now embrace, With perfume anointed, of a comely grace, For whom thou trimmest up thy yellow hair Only with this desiring him to cheer. Alas! How oft shall he, poor soul, deplore That not performed which thou hadst sworn before. O how shall he admire the calmest seas, On sudden void of pleasure, rest, and ease, And thee, before which on his will didst wait, To be inconstant and quite changed of late, Embracing thee as constant, firm, and true, Not knowing that thy faithless deeds ensue. O woeful men that are by thee ensnared! But me the gods from thy false wiles have spared, And I do offer up to Neptunes shrine My garments pure, and neer defiled by thine Untrue, deceitful, famed inconstancy, As men from shipwreck that have escaped free.",
        "author": "William Ainsworth",
        "date": 1625
    },
    {
        "text": "What tender boy upon a rosie bed, Being with liquid odours overspred, Within some pleasant bower, doth to thee sue (O Pyrrha) for thy love? for whom doe you Bind your gold locks, plain in your ornament? Alas, how often shall the proud Boy repent Thy false faith, and condemned deities, And look with wonderment on those thy seas Made rough with black winds, who (too credulous Boy) Does thee now as some golden prize enjoy? Who hopes thoult still be free to him, still faire, Ignorant of thy all-deluding aire. Wretched are they to whom untride you shine; The wall, by sacred tables made divine, Shewes I have hung my ship-rackt robe on high Unto the Oceans potent Deitie.",
        "author": "Henry Rider",
        "date": 1638
    },
    {
        "text": "Tell me, Pyrrha, what fine youth, All perfumed and crowned with roses, To thy chamber thee pursues, And thy wanton arm encloses? What is he thou now hast got, Whose more long and golden tresses Into many a curious knot Thy more curious finger dresses? How much will he wail his trust, And, forsook, begin to wonder, When black winds shall billows thrust, And break all his hopes in sunder? Fickleness of winds he knows Very little that doth love thee; Miserable are all those That affect thee ere they prove thee. I, as one from shipwreck freed, To the oceans mighty ranger Consecrate my dropping weed, And in freedom think of danger.",
        "author": "William Browne",
        "date": 1615
    },
    {
        "text": "What slender youth is he that oft doth meet With Roses, and moist odours smelling sweet, Under thy grateful Cell, O Pyrrha faire? To whom dost thou bind up thy yellow hair? In outside simple: alas, how in vain Shall she of love and changed gods complain, To see smooth Seas, with black winds soon made rough, Nor being skillful shall complain enough; Trusting that he alone enjoys thy love From all; and hopes it never will remove: Poor simple soul, not knowing thy false heart, Now woe to whom thou smites with unknown Art, The votive picture to the wall made fast, Doth shew that I my moist clothes have at last Hung up to thee, O Neptune god of Seas, Since me to save from shipwreck thou didst please.",
        "author": "John Smith",
        "date": 1649
    },
    {
        "text": "What Stripling now Thee discomposes, In Woodbine Rooms, on Beds of Roses, For whom thy Auburn Hair Is spread, Unpainted Faire? How will he one day curse thy Oaths And Heaven that witnessed your Betroths! How will the poor Cuckold, That deems thee perfect Gold, Bearing no stamp but his, be amazed To see a sudden Tempest raised! He dreams not of the Winds, And thinks all Gold that shines. For me my Votive Table shows That I have hung up my wet Clothes Upon the Temple Wall Of Seas great Admiral.",
        "author": "Richard Fanshawe",
        "date": 1652
    },
    {
        "text": "What spritely Younker amongst Beds of Roses, (Pyrrha) perfumed with fragrant scents incloses Thee skulked in sweet retire? Thy fair locks, at whose desire Pleatst thou so up, arrayed in homely clothes? O, how hell wail thy oft-changed Gods, and oaths, And count it wondrous strange When storms in thy countenance range! To whom thou now vouchsafes a Golden Grace, Hoping you’ll still find leasures for embrace, And constantly be kind, Not versed in thy crafty mind. O cursed are they who trust thy shining hew! I hung (as Votive frames in Temples shew) Moyst robes up to appease Neptune, powerful God of Seas.",
        "author": "Barten Holiday",
        "date": 1653    },
    {
        "text": "To whom now Pyrrha art thou kind? To what Heart-ravisht Lover Dost thou thy golden locks unbind, Thy hidden sweets discover, And with a large bounty open set All the bright stores of thy rich Cabinet? Ah simple youth, how often will he Of thy changed faith complain? And his own fortunes find to be So airy and so vain, Of so Chameleon-like an hew, That still their colour changes with it too. How often alas will he admire The blackness of the skies? Trembling to hear the winds sound higher, And see the billows rise, Poor inexperienced he, Who neer before alas had been at Sea! He enjoys thy calmy Sun-shine now, And no breath stirring hearts; In the clear heaven of thy brow, No smallest cloud appears; He sees thee gentle, fair and gay, And trusts the faithless April of thy May. Unhappy! thrice unhappy he, To whom thou untried does shine, But theres no danger now for me, Since over Loretto’s shrine, In witness of the shipwreck past My consecrated vessel hangs at last.",
        "author": "Abraham Cowley",
        "date": 1666    },
    {
        "text": "What slender Youth bedewed with liquid odours Courts thee on Roses in some pleasant Cave, Pyrrha for whom bindest thou In wreaths thy golden Hair, Plain in thy neatness; O how oft shall he On Faith and changed Gods complain: and Seas Rough with black winds and storms Unwonted shall admire: Who now enjoys thee credulous, all Gold, Who always vacant, always amiable Hopes thee; of flattering gales Unmindful. Hapless they To whom thou untried seems fair. Me in my vowed Picture the sacred wall declares to have hung My dank and dropping weeds To the stern God of Sea.",
        "author": "John Milton",
        "date": 1673    },
    {
        "text": "What mean those Amorous Curls of Jet? For what heart-Ravished Maid Dost thou thy Hair in order set, Thy Wanton Tresses Braid? And thy vast Store of Beauties open lay, That the deluded Fancy leads astray. For pity hide thy Starry eyes, Whose Languishments destroy: And look not on the Slave that dyes With an Excess of Joy. Defend thy Coral Lips, thy Amber Breath; To taste these Sweets lets in a Certain Death. Forbear, fond Charming Youth, forbear, Thy words of Melting Love: Thy Eyes thy Language well may spare, One Dart enough can move. And she that hears thy voice and sees thy Eyes With too much Pleasure, too much Softness dies. Cease, Cease, with Sighs to warm my Soul, Or press me with thy Hand: Who can the kindling fire control, The tender force withstand? Thy Sighs and Touches like winged Lightning fly, And are the God of Loves Artillery.",
        "author": "Aphra Behn",
        "date": 1684    },
    {
        "text": "What tender Youth upon a Rosy bed, With Odours flowing round his Head, Shall ruffle Thee, and lose a heart? For what fond Youth wilt Thou prepare The lovely Mazes of thy Hair And spread Charms neat without the help of Art? How often unhappy shall he grieve to find The fickle Baseness of your Mind? When he, that neer felt storms before, Shall see black Heaven spread oer with Clouds, And threatening Tempests toss the Floods Whilst helpless He in vain looks back for Shore. Now fondly, now He rifles all thy Charms, He wantons in thy pleasing Arms, And boasts his happiness complete: He thinks that You will always prove As fair, and constant to his Love; And knows not how, how soon those smiles may cheat. Ah! wretched those who love, yet neer did try The smiling treachery of thy Eye! But Im secure, my Dangers oer, My Table shows the Clothes I vowed, When midst the storm, to please the God I have hung up, and now am safe on shore.",
        "author": "Thomas Creech",
        "date": 1684    },
    {
        "text": "What slender Youth (Pyrrha) with Roses, Whose anointed Head perfumes discloses? Clasped thee in a secret place; Whilst Beam-like Hair, with loosened grace, Doth simple cleanliness present; How oft, alas, hel mourn, lament? Thy Vows, Gods changd, and lost; Wondering to find calm Seas ore-tossed, Who credulous believes thee now All Gold, still hopes for pleasing Brow: Bad-learnd in falser Air: Those whom thy Looks (untried) ensnare Accurst; my self wrack scaping so, Let powerful Sea-gods Temple show On the Wall my Thanks, where now Wave-drenched Clothes hang up for vow.",
        "author": "John Harington",
        "date": 1684    },
   {
        "text": "Pyrrha, what young tender Boy Shall thee on rosie Beds enjoy? Tell me, O tell me, Charming Fair, For whom you spread your Golden Hair? To whom do you display your Charms, And all your Beauties naked lay? What Youth is to be happy made Beneath this grateful pleasant Shade? Alas! how will he curse the Charms, That led him Captive in thy Arms? How will he curse thy broken Vow? And Heaven that did thy Oaths allow? When he expected still to find New Joys in thee to please his Mina, And that thou always wouldst be kind. When he thought all thy Charms Divine, No Beauty and no Truth like thine: Not dreaming of a Tempest, when He saw all clear and all serene: O how amazed the Wretch will be, When by thy Falsehood he shall see A sudden dark tempestuous Sea? Mistaken Fool, who never before, Had ventured from his native Shore; For Me, who have your Falsehood known, My Votive Table shpws Ive done: That Ill no more believe thy Smiles, Thy Sighs, and Tears, and winning Wiles; But keep my self secure ashore, And trust the Winds and Waves no more.",
        "author": "Robert Clavering",
        "date": 1707    },
   {
        "text": "Pyrrha, what slender well-shaped Beau, Perfumed with Essence haunts thee now, And lures thee to some kind Recess, To Sport on Rose-Beds sunk in Ease? Prithee what Youth wouldst thou insnare, Artless and clean, with flowing Hair? How often will he have cause to mourn Thy broken Vows and Cupids Scorn? Unskilled as yet, hed wondering spy Fresh tempests raging in that Eye, From whence he hoped a Calmer Sky, Who now poor Gull enjoys the Bliss, Thinks you divine and solely his: Born down the Tide with easie Sail, Little suspects an Adverse Gale. Thrice wretched they who feel thy Darts, Whilst Strangers to thy Coquet Arts! My Garments in the Fane displayed, As Trophies that my Vows are paid, Own the Great Ruler of the Sea Author of my Delivery.",
        "author": "Philip Horneck",
        "date": 1708    },
   {
        "text": "What well-shaped Lover in the Rosie Shade, With fragrant Limbs and Sweet Address, Shall to thy warm Embraces press, In all thy loose Attire and wanton Airs displayed? Bright Charmer, nicely clean tho Plain! How shall the Youth with sad Surprise, See angry Storms and Tempests rise, And all this Calm of Love break into fierce Disdain? He doats, he raves with Bliss, whilst thou art kind; Ah Wretch! undone by Amorous Smiles, Who sees thy Charms and not thy Wiles; For thou art light as Air, inconstant as the Wind. Learn from my Fate; by Tides and Whirlwinds tost, I reached the shore, half-drowned in Brine; My tablet hangs on Neptunes Shrine, To warn all other Sailors from the dangerous Coast.",
        "author": "William Oldisworth",
        "date": 1713    },
{
        "text": "What Stripling, Pyrrha, what Perfumed Boy, Is thy lewd Art contriving to decoy; Enticed by thee to some lone Cave, or Grove, Fit for thy private Purposes of Love? For whom dost thou in sober Dress appear, With clean, and modest Tire to thy Hair? How often, alas! will the poor Captive grieve, That does thy promised Constancy believe? When he perceives that pleasant smiling Face Change, and look Frowning, as the angry Seas, And Venus, and her Son, depart and fly, And all the Gods quite Vanishd that were nigh. Unhappy, and too Miserable He, That inexperienced sets his Love on Thee! To Neptune I have offered up my Store, Since I escaped Shipwreck on thy Shore.",
        "author": "Henry Coxworth",
        "date": 1718    },
{
        "text": "What Stripling essenced, in some cool Recess, Does, Pyrrha, thee on Beds of Roses press? For whom dost thou, with such an easy Air Genteely dressed, adjust thy Golden Hair? Poor inexperienced Youth! hell quickly find His Tide of Pleasure ruffled by the Wind! How often shall he that now enjoys thee, say, How Pyrrha differs from herself to Day! For in his next Adventure, full of Joy, You’ll prove engaged, or more surprising, coy. Woe be to them, whom your bright Charms allure, And think themselves before they’ve tried you, sure! My dripping Jacket, which you might have wrung, Sacred to Neptune, on his Wall I've hung.",
        "author": "John Hanway",
        "date": 1720    },
{
        "text": "Come, Pyrrha, tell what lover now Is most in your good graces? On what laced coat, or scented beau, In public you your smiles bestow, And more in private places. What easy heart do you invade By all this nice adorning? For what vain fop is now displayed: The mechlin lace and rich brocade, At toilet spent the morning? Ah! how hell rage, when midst his calm Tempestuous clouds shall gather, When he beholds the lowering storm, That faithless brow of thine deform, Untried in boisterous weather! Whom now thy look serene beguiles, Ah poor, unthinking creature! Who credulous, enjoys thy smiles, And never dreaming of thy wiles, Now thinks thee all good-nature. He feels thy charms in a wretched hour, Thats to thy ways a stranger: As for my part, my turn is over, Ive escaped the deep, and safe from shore, Look on anothers danger.",
        "author": "James Ward",
        "date": 1720    },	
{
        "text": "What slender youth, my Pyrrha, now Dropping with odour’s liquid dew, Thee to the grateful cave invites And urges to the soft delights, Where flow’rs the bed of love compose, The down profuse of many a rose? To whom thy length of yellow hair Thou ty’st behind, t’express Sweet negligence of artful care, Plain in virginity of dress? How oft shall he with wat’ry eyes Lament the alienated skies, Thy broken faith, and to his cost The beauty of the ocean lost Admire, a stranger to the sight, Rough with storms and black with night? Who now possesses thee, all gold, Who, fond believer, hopes to hold Still vacant to himself, and free, Ever in sunshine and serene; Of flattering gales unskilful he That may disturb the smiling scene. Miserable! When untried To whom thou shines; escaped the tide, The sacred Tablet speaks for me, High on the votive wall along My garment dropping wet to have hung To the potent God of sea.",
        "author": "William Hamilton of Bangour",
        "date": 1720    },	
{
        "text": "What well-shaped youth, false Pyrrha, say, Perfumed with Essence, sweet and gay, In the cool Grotto of the Grove, Upon the fragrant rosy Bed, On thy fair Breast reclines his Head, And breathes his ardent Love; Tell me, for whom dost thou neglect (Too skillful in the Art to please) Thy golden Hair? For whom affect That elegant Simplicity of Dress? Alas, the easy, credulous Youth Thinks thee all Constancy and Truth; Untouched by any ruffling Blast, Trusts in the Summer of thy Smiles, With the fair Show himself beguiles, And thinks the Calm will last; But soon the furious Storms shall rise, And then the wretch shall start to find A boisterous wind, and stormy Skies, And wonder at perfidious womankind. Unhappy they whose witless Hearts, Unpracticed in thy subtle Arts, Are doomed to drag thy tyrant Chain; Myself have known thee, to my Cost, And once I gave myself for lost, But now respire again I bless the gracious Powers above That saved me from the Witchcraft of thy Love.",
        "author": "Glocester Ridley",
        "date": 1720    },	
{
        "text": "What stripling Youth, in some convenient Shade, Does Pyrrha tempt, on Beds of Roses laid? For whose Reception does the Nymph prepare, That something, so genteel, in Dress and Air? Poor inexperienced Youth! hell quickly find, His Tide of Pleasure ruffled by the Wind; For in his next Adventure, full of Joy, Shell prove engaged, or (more surprising!) coy. Thank my kind Stars Im safely got to shore, Resolved to trust that treacherous Calm no more.",
        "author": "Anonymous, British Journal",
        "date": 1725    },	
{
        "text": "Who, Amorett, is now the Joy Of thy fond heart? what blooming Boy, Rich-essenced, and on Rose-beds laid, Pants over thee in the Grottos Shade? For whom, like Rural Maidens fair, Wreaths thou with Flowers thy flaxen Hair? How oft shall He thy Faith arraign? Of the changed Gods how often complain? With what surprise, unwont, survey The lowering Heavens and clouded Day? The Youth, who, now with Smiles carest, Trusts in the Charms, that make him blest: Who paints thee vacant, lovely, kind; Unweening of the faithless Wind! Cursed! who to those false Smiles confide; Doat on that darling Face untried! In yonder Tablet it is expressed, That I have hung my Sea-dank Vest, An Offering, in his sacred Shrine, To the great Power, that rules the Brine.",
        "author": "Leonard Welsted",
        "date": 1725    },	
{
        "text": "What Youth in rosy Bower laid, His Locks with liquid Odours spread, Now hugs thee to his panting Breast? And thinks no Mortal half so blest? For whom dost thou, enchanting Fair, In Ringlets wreath thy flowing Hair? For whom, my Pyrrha, dost thou deign To deck thus elegantly plain? The unwary Wretch, who sees no Guile, Drinks Poison in at every Smile, And figures to his flattering Mind Thee, always vacant, always kind; Unwont to see, unwont to hear One chiding Word, or Look severe; How shall he view, with secret Dread, That heavenly Face with Clouds overspread? How often curse his fatal Love? His gods? who so inconstant prove. Ah, hapless they! who view that Face Adorned with every winning Grace; Unknowing Pyrrhas fickle Heart Full fraught with all-deceiving Art. In yonder votive Tablets read How I, from dreadful Ship-wreck freed, My dropping Weeds hung up to Thee, Great Neptune, Ruler of the Sea.",
        "author": "Eugenio",
        "date": 1729    },	
{
        "text": "On Beds of Roses, Pyrrha, say? What well-shaped Youth perfumed with Oil, In some cool Grott, Excluding-Day, Urges thee close with amorous Toil? For whom thus careless knot thy Hair? More charming in thy want of Care! How oft shall be, ill-fated Youth! With Tear regret thy Breach of Truth? Thy faithless Gods! and Sea, at last, By Tempests and black Winds overcast! Which he, unused, with Wonder shall behold! Who, credulous, thinks thee now like Gold! Hoping, thou (vain Hope!) wilt be Ever lovely, ever free! Fond Youth! He little knows The treacherous Wind that blows! Hapless they! that in their Arms, Clasp thy unexperienced Charms! escaped from the Storm, and safe ashore, In Neptunes Shrine Ive hung my Clothes; There too, the Votive Picture shows, Once Ship-wrecked, Ill go to Sea no more.",
        "author": "William Popple",
        "date": 1735    },	
{
        "text": "Say what dear Youth his amorous rapture breathes Within thy arms beneath some Grotto reclined? Pyrrha, for whom dost thou in wreathes Thy golden tresses bind In plainness elegant? how often shall he Complain alas! upon the fickle skies And suddenly astonished see The blackening tempest rise: Who now enjoys thee, happy in Conceit Who fondly thinks thy love can never fail Never to him– unmindful yet Of the fallacious Gale. Wretch! to whom thou untried seemest fair, For me, Ive escaped the Wreck; let yonder fane Inscribed my gratitude declare To him that rules the Main.",
        "author": "Richard West",
        "date": 1737    },	
{
        "text": "What shapely Beau on Roses laid Rich Odours breathing from his Head, Beneath a secret, sweet Alcove, Invites thee, Pyrrha, to his Love? For whom, fair Tempter, does thou bind Thy golden Locks in Wreaths behind? For whom engagingly express That simple Elegance of Dress? How often alas! shall he complain Of adverse Gods, Vows made in vain; And view the Sea with vast Surprise In lowering Storms tremendous rise; Who now enjoys thee, charming Fair, A Stranger to thy flattering Air; Who fondly hopes he still shall find Thee disengaged, and gently kind. Wretched are those thy Face beguiles, Who never have proved your treacherous Smiles; I just have escaped the raging Brine, And (joyous) hung my Weeds at Neptunes Shrine.",
        "author": "Thomas Hare",
        "date": 1737    },	
{
        "text": "While liquid odours round him breathe, What youth, the rosy bower beneath, Now courts thee to be kind? Pyrrha, for whose unwary heart Do you, thus dressed with careless art, Your yellow tresses bind? How often shall the unpractised youth Of altered gods, and injured truth With tears, alas! complain? How soon behold with wondering eyes The blackening winds tempestuous rise, And scowl along the main? While by his easy faith betrayed, He now enjoys thee, golden maid, Thus amiable and kind; He fondly hopes, that you shall prove Thus ever vacant to his love, Nor heeds the faithless wind. Unhappy they, to whom untried You shine, alas! in beautys pride; While I, now safe on shore, Will consecrate the pictured storm, And all my grateful vows perform To Neptunes saving power.",
        "author": "Philip Francis",
        "date": 1743    },	
{
        "text": "What Youth perfumed with liquid Sweets, Genteely drest, thee Pyrrha greets, On Beds of fragrant Roses laid, Beneath a pleasant secret Shade? For what Gallant dost Thou prepare The golden Ringlets of thy Hair: Artlessly neat? How will the Swain Of thy Inconstancy complain; And faithless Gods! with what Surprise Must he behold the Storms arise; Who now enjoys thy easy Smile, And thinks it never can beguile; Who hopes alone to clasp thy Charms, And find thee yielding to his Arms, Unknowing thy deceitful Airs! Wretched, who cannot see thy Snares, Yet trust thy Face. The Temple shows My late Escape, and finished Vows.",
        "author": "The Museum",
        "date": 1747    },	
{
        "text": "Whence this triumphant smile you wear? Why this gay dress? this flowing hair? Say, say, what graceful swain, Proud to endure the pleasing pain, Seeks the dark covert of the grove, To pour a melting tale of love. Unhappy youth! though now he feel A joy that words can never reveal, And fondly hopes in thee to find A heart still constant, and still kind, Enchanted by thy heavenly form, And thoughtless of the impending storm, How soon astonished, shall he see Thy broken faith, thy perjury, And curse that more than fatal day, Which gave his willing soul away! For me, the tempest safely over, With joy I tread the friendly shore, Bless the kind power that set me free, And triumph in my liberty: Nor Miras self shall make me prove, Anew, the boisterous sea of love.",
        "author": "The Gentleman’s Magazine",
        "date": 1748    },	
{
        "text": "For whom are now your Airs put on? And what new Beauty doomed to be undone? That careless Elegance of Dress, This Essence that perfumes the Wind, Your every motion does confess Some secret Conquest is designed. Alas the poor unhappy Maid, To what a train of ills betrayed! What fears! what pangs shall rend her Breast! How will her eyes dissolve in Tears! That now with glowing Joy is blessed, Charmed with the faithless vows she hears. So the young Sailor on the Summer Sea Gaily pursues his destined way, Fearless and careless on the deck he stands, Till sudden storms arise, and Thunders roll, In vain he casts his Eye to distant Lands, Distracting Terror tears his timorous Soul. For me, secure I view the raging Main, Past are my Dangers, and forgot my Pain, My Votive Tablet in the temple shews The Monument of Folly past, I paid the bounteous God my grateful vows Who snatched from Ruin saved me at the last.",
        "author": "Lady Montagu",
        "date": 1750    },	
{
        "text": "On beds of Roses, Pyrrha, say, In some cool Grot, excluding Day; What well-shaped Youth, perfumed with Oil, Urges thee close, with amorous Toil? For whom ties thou thy golden hair? Still charming in neglect of care! How oft shall he (ill-fated Youth!) With Tears regret thy breach of Truth? Thy faithless Gods, and Sea at last, With Tempests and black Winds o’ercast; Which he who thinks thee pure as Gold, Too credulous shall soon behold! Believing Thou (vain hope!) will be, Still lovely, disengaged and free! Alas! fond Youth! he little knows, The smooth, but treacherous Wind, that blows! Unhappy those, who in their Arms, Shall clasp thy unexperienced charms! Suspended in great Neptune’s Fane, My Clothes and Picture now remain: A monument, that safe on shore, Thy Lover will to Sea no more.",
        "author": "William Popple",
        "date": 1750    },	
{
        "text": "What essenced youth on beds of roses laid Courts thee, O Pyrrha, in some pleasing shade, In artless elegance of dress arrayed? What favourite swain commands thy nicest care, And bids those ringlets grace thy flaxen hair? Oft of the Gods he’ll with a sigh complain, Often at your broken vows lament in vain: Secure and heedless of the shifting scene, Surprised he’ll view that aspect once serene Ruffed with frowns; who now within his arms Thinks he possesses you in all your charms; And too too easily believes you’ll prove True to his passion, yielding to his love. Wretched are those, whom that fair form beguiles, Dupes to your charms, and victims of your smiles! My weeds all wet and dripping from the main, And votive tablet hung on high proclaim My bliss secure, and peace restored again.",
        "author": "The Oxford Monthly Miscellany",
        "date": 1750    },	
{
        "text": "What slender Boy, with Odours sweet, Shall in a Grotto’s cool Retreat, Thy too-enchanting Form caress, And on a couch of Roses press? For whom in Wreaths dost thou prepare, So simply neat, thy golden Hair? How oft, of Gods adjured in vain, And broken Vows, shall he complain! How oft admire, when Winds arise, To see black Clouds deform the Skies; New to the Sex, who tastes thy Charms, And fondly clasps thee in his Arms; In thee a Mistress ever kind, And ever lovely, hopes to find; And thinks, too credulous, the Breeze Will last; nor Tempests toss the Seas! Ah wretched they! whom Pyrrha’s Smile, And unsuspected Arts beguile! For Me, the sacred Tablet shows That I have hung my dripping Clothes At Neptune’s Shrine: And now on Shore Secure, I’ll tempt the Deep no more.",
        "author": "William Duncombe",
        "date": 1757    },	
{
        "text": "Say, Pyrrha, who’s this slender boy, Deep in love and amorous joy, Who upon a fragrant bed, Late of new-blown roses made, In the grotto’s pleasing shade, Dank yet with his liquid sweets, Kiss after kiss, enamoured thus repeats? For whom thus do you braid your hair Carelessly neat, and thus appear Gay in simplicity? poor swain, How oft, alas, will he complain Of vows and plighted gods in vain! How oft, alas, he’ll curse his fate, When all your perjury’s found out too late! How will he then indulge surprise, When once he sees the tempest rise! When he shall view that once-smooth mind, Foaming with rage, and as unkind As the rough sea with boisterous wind! Stupid in wonder he’ll admire, Such falsehood foul under such fair attire. Who now enjoys your dear-bought charms, Encircled in his longing arms, Who hopes you’ll ever be sincere To him alone, and always dear; Not knowing yet his fatal snare. Ah! wretched to extremity, Who experienced are allured by thee. Since I’ve escaped clear, and got to shore, Finding myself set free once more,I have (according to my vow) Hung up my tablet, which may shew The case I was, and am in now. My clothes hang dripping in the fane, Of Neptune, that great ruler of the Main",
        "author": "Elizabeth Carolina Keene",
        "date": 1762    },	
{
        "text": "By what smart beau, with liquid nard bedewed, In beds of roses, in a cool alcove, Art thou, incomparable Pyrrha, wooed, In all, the wild extravagance of love? For whom in wanton ringlets dost thou tie The shining mazes of thy golden hair, Formed to engage each fond beholder’s eye, In unaffected delicacy fair How oft, alas! shall he, in wild amaze, Of broken vows and fickle gods complain, And stand aghast when sudden winds shall raise The roughening waves of the late placid main? Who thoughtless now thy venal charms enjoys, And hopes thee ever disengaged and kind; By slatt’ring gales betrayed, and treacherous skies, Shall wonder such unwonted gusts to find. Unhappy they, and born to curse their fate, Who, ravished with thy negligence of art, Too blindly love, nor e’er suspect deceit, But think thy face the image of thy heart! I, who escaped the danger of the main, And landed safely on the wished-for shore, My dropping weeds suspend in Neptune’s fane, On tablets vowed, in reverence to his power.",
        "author": "Samuel Rogers",
        "date": 1764    },	
{
        "text": "Say what slim youth, with moist perfumes Bedaubed, now courts thy fond embrace, There, where the frequent rose-tree blooms, And makes the grot so sweet a place? Pyrrha, for whom with such an air Do you bind back your golden hair? So seeming in your cleanly vest, Whose plainness is the pink of taste Alas! how oft shall he protest Against his confidence misplaced, And love’s inconstant powers deplore, And wondrous winds, which, as they roar, Throw black upon the altered scene Who now so well himself deceives, And thee all sunshine, all serene For want of better skill believes, And for his pleasure has presaged Thee ever dear and disengaged. Wretched are all within thy snares, The inexperienced and the young! For me the temple witness bears Where I my dropping weeds have hung, And left my votive chart behind To him that rules both wave and wind.",
        "author": "Christopher Smart",
        "date": 1767    },	
{
        "text": "Say, Pyrrha, what enraptured boy, On yonder rosy couch reclined, Thy charms now presses to enjoy, Profuse of odours to the wind? Tell me the youth, at whose desire Your beauteous auburn locks you braid; For whom in simply neat attire You act the fond complying maid? Ah! treacherous, he shall soon perceive Your slighted vows and cold disdain: The winds rise high, the billows heave: He seeks for azure skies in vain. The youth, who rifles all your charms, Too soon shall see you with surprise Expiring in another’s arms, And own the treachery of your eyes. For me the votive tablets shew, Preserved from shipwreck safe on shore, To Venus’ train I bid adieu, And launch into the sea no more.",
        "author": "C.D., The London Magazine",
        "date": 1769    },	
{
        "text": "Pyrrha, what slender pretty boy, Bedewed with rose abundant prest, Dost thou to pleasing grot decoy? Who next to be caressed? For whom thy knotted golden tress; In neat simplicity of grace, So elegantly plain? How soon, how soft, shall he complain Of shifting gods, and fickle troth, Unwonted wondering Youth? When o’er the skies serene, The fable, angry clouds arising, And sudden squally forms surprising, Scowl along the main? Deluded, hapless boy, Vain hoping pure of all alloy, For ever vacant to his arms, Forever amiable, all charms The melting golden joy. Too credulously blind, To tempest imminent, Unknowing in the element, Of the fallacious wind. Ah wretched he, to whom untried, Thou glitterest– I the swelling tide, Escaped thank Neptune, safe on shore. My votive tablet points to all My dripping garments, on the wall, Suspended– to the saving power.",
        "author": "William Green",
        "date": 1777    },	
{
        "text": "Who’s the handsome young stripling so happy presumed, That with essence of sweet-scented odours perfumed, To the rose-paved grotto resorts, And, Pyrrha, thy company courts? To engage him you ringlet your beautiful hair, And with natural neatness to enamour prepare; But how often, alas, shall he mourn That fortune and faith backward turn! As when suddenly rough, with black tempest, appears All the sea, the unaccustomed passenger stares; So shall he who, too credulous now, A treasure possesses in you, Disengaged still and lovely you hoping to find, Not aware how deceitful is trusting the wind: Woful must be their case who confide In beautiful Pyrrha untried! On a plate vowed in peril the inscriptions declare; On the wall consecrated, that my garments there Were all dripping hung up, to display His goodness who governs the sea.",
        "author": "John Gray",
        "date": 1777    },	
{
        "text": "What slender paramour under a rosy cave Courts thee, sweetly bedewed with liquid essences? Say, fair Pyrrha, for whom thou Binds thy tresses of wavy gold, In plainness elegant? Often, alas! will he Weep, and fondly bewail thy mutability, Oft, rough with many tempests, View yon seas with astonishment, Who now, credulous youth, folds thee in ecstacy, Who thee, ever a kind, ever a lovely maid Hopes, unmindful of breezes Fallacious! O unhappy, whom Thy strange beauty delights! Me on the holy wall Yon votive monument indicates here to have Hung my watery vestments To the stern God of Ocean.",
        "author": "Sir William Jones",
        "date": 1781    },	
{
        "text": "Pyrrha, what slender pretty boy, Bedewed with fragrant roses prest, Is now in thy false arms caressed, In the delicious grot of joy? For whom thy knotted golden tress, In neat simplicity of grace Dost bind, so elegantly plain? Of violated faith and truth, And changing Gods, unwonted Youth, How soon, how oft shall he complain, When over the face of Heaven serene, Sudden, he views with wondering eyes, The sable squally austers rise, And scowling sweep along the main, Portending hurricanes and rain? Who holds thee now delighted boy, He, bright and pure of all alloy, For ever amiable all charms, And solely vacant to his arms, Vain hopes the melting golden joy. Deluded, credulously blind, Unskilled in the fallacious wind. Ah! wretched he, to whom untried, Thou glitterest– I the swelling tide, Thank Neptune, escaped in happy hour, The votive tablet points to all My dripping garments, on the wall, Suspended– to the saving power.",
        "author": "William Green",
        "date": 1783    },	
{
        "text": "Pyrrha! who’s the youth, that now, Odours dropping from his brow, Woos thee in the tender hour All beneath the rosy bower? For whom dost thou with treacherous care Now teach thy lovely golden hair With all the easy grace to lie Of elegant simplicity? How oft alas shall he deplore, The Gods propitious now no more! And curse the dear deluding Maid And his fond heart so soon betrayed! And see with wild, astonished eyes Sudden storms around him rise, And angry winds with fury sweep The foaming billows of the deep! Who now, alas! mistaken boy! Drunken with excess of joy, On thee dotes the live-long day Ever kind and ever gay, And to all thy frailties blind, Hopes, too fondly, thee to find Still as gay, and still as kind. Ah! the pangs, that they must prove Who rashly hope that thou canst love! See! my votive tablet shews, Faithful picture of my woes, Consecrate, dread power! to thee, Mighty monarch of the sea! And my weeds all dank and wet Tattered all and dripping yet, Shew that I, escaped at last, Smile secure at dangers past.",
        "author": "William Lipscomb",
        "date": 1784    },	
{
        "text": "Seductive Pyrrha! what enamoured boy, In whose bright locks the liquid odour flows, Woos thee? and triumphs in a short-lived joy Within the grotto, adorned with many a rose! For whom, in ample grace, dost thou prepare The Band that lightly ties thy golden hair? Alas, how soon shall this devoted youth Love’s tyrant sway and thy changed eyes deplore! Indignant curse thy violated truth, And count each broken promise over and over, Who hopes, unconscious of thy fatal wiles, A long duration of those lovely smiles! He, inexperienced mariner! shall look In wild amazement on the stormy deep, Who, when his heedless bark the port forsook, Had lulled each ruder wind in softest sleep; ’Twas then he fondly spread the swelling sail, In rash dependance on the faithless gale. Ah wretch! to whom untried thou shinest fair! By me, who late thy glassy surface sung, The walls of Neptune’s fane inscribed, declare That I have dank and dropping garments hung, Devoted to the God, whose kind decree Snatched me to shore from an overwhelming sea!",
        "author": "Anna Seward",
        "date": 1786    },	
{
        "text": "What graceful youth, whom liquid sweets bedew, Now courts thee willing in some pleasant bower, Where the fair rose spreads round her flaunting flower, And sheds a crimson couch? for whom do you, Pyrrha, now braid your hair of golden hue, In neatness plain? How oft shall he deplore Thy changed faith, and when the black winds roar, With watery eye the swelling billows view, Who credulous enjoys thee, precious now, Who hopes thee vacant still, still smooth thy brow. Poor wretch! of flattering gales unmindful he? Luckless are they, who, all unweeting, thee Admire– Me doth the votive tablet show, To have hung my drenched weeds to the God of Sea.",
        "author": "The Gentleman’s Magazine",
        "date": 1787    },	
{
        "text": "What youth, laid on a rosy bed, With odours flowing round his head, In a cool grot does you caress? For whom do you, deluding Fair! Adjust your head, and plait your hair, And so genteely dress? Alas! how often will he find The various, motions of your mind Unsettled, unsedate! View frowns, subservient to your wiles, Supplant your momentary smiles; And curse his cruel fate! Who now enjoys at large your charms, And, melting in your circling arms, Believes your love sincere? The hope, you ne’er will from him part, Foments the passions of his heart, And banishes despair. Unhappy youths! unhappy they, Your unsuspected arts betray! In Neptune’s temple, view A votive tablet and a vest, By me suspended as a test Of my escape from you.",
        "author": "Gilbert Wakefield",
        "date": 1790    },	
{
        "text": "Who, fair Pyrrha, wins thy graces? What gay youth imprints a kiss? Or in roseate groves embraces Urging thee to amorous bliss? To delude to your caresses What young rake, or wanton blade, Do you bind your golden tresses, In plain elegance arrayed? Soon the unhappy youth, deploring, Shall lament thy proud disdain; Thus, the winds, tempestuous roaring, Rend the bosom of the main. He, who’s now thy beauty prizing, In thy smiles supremely blest, Dreams not of the storm that’s rising, To disturb his peaceful breast. Misery’s sharpest pang he suffers, Who, secure from all alarms, Like all thy deluded lovers, Clasped a serpent in his arms. Once, thy deep intrigues unknowing, I embarked upon the deep; Boisterous storms, dread horrors blowing, Roused me from lethargic sleep. Billows were around me roaring, When great Neptune’s friendly aid, Me to Rome again restoring, There my grateful vows I paid.",
        "author": "Robert Treat Paine",
        "date": 1790    },	
{
        "text": "What youth, laid on the flowery ground, With grateful odours flowing round, Now courts you to be kind? For whom, insidious, charming fair! Do you the tresses of your hair In wanton ringlets bind? Alas, how often shall he prove You’re false and faithless to his love, And all your vows untrue! One moment he shall see you kind, The next as fickle as the wind, And curse his love and you. While now the unsuspecting boy, Dissolved in bliss, does you enjoy, And hopes you’ll constant prove: Ah, wretched youth! the varying skies Will change, and dreary tempests rise, And blast your fondest love! But know, too false, alluring fair! Thy charms no more shall me ensnare, To them I bid adieu! In Neptune’s sacred temple see The trophies of my victory, Obtained over Love and You!",
        "author": "R Lickorish",
        "date": 1790    },	
{
        "text": "What graceful youth, on rosy beds reclined, My Pyrrha woos thee in some pleasant cave? For whom dost thou thy golden tresses bind, That to the gale in simple beauty wave? When ether blackens, and the furious main Tremendous winds and dashing waves deform, How often will he of faithless gods complain, While all amazed, he sees the unwonted storm! Whose easy hopes, by golden views deceived, Still paint thee ever lovely, ever kind, Their syren whispers vainly are believed, Still art thou false, and fickle as the wind. O hapless they! to whom thy faith, untried, Seems fair and spotless. Me, now safe on shore, My tablet shews, on Neptune’s altar tied, My dropping weeds, and offering to his power.",
        "author": "J.B. The Universal Magazine of Knowledge and Pleasure",
        "date": 1790    },	
{
        "text": "What taper stripling now bedews His locks with liquid sweets, and wooes Thee, underneath the arched shade, Pyrrha, on fragrant roses laid, For whom you bind your flowing hair With that sweet, simple, graceful air? Alas, how often shall he in vain Of broken vows and Heaven complain! And when he views the angry deep Grow black with winds, astonished weep, Who now with thee the golden hour Enjoys, and smiles at Fortune’s power? Thee ever constant, ever kind, Fond, easy youth! expects to find, Nor dreads the treacherous storm behind. Ah! wretched they, on whom benign Thy fickle graces newly shine! For me, no more I tempt the main; High on the wall of Neptune’s fane, My escapes, my perils, and my woes, My votive tablet duly shews. Saved, to his Guardian Power I bend, To him my dripping weeds suspend.",
        "author": "Sagittarius, Gentleman’s Magazine",
        "date": 1790    },	
{
        "text": "Ah! tell me, dear Pyrrha, what beautiful boy, This evening shall rifle these charms; Some jessamine arbour the scene of your joy, And Paradise all in your arms! For whom are you combing your long jetty hair, So gracefully artless your dress; So tender a look! so bewitching an air! Admiration swells into distress. Your simple young favourite will fondly suppose, That he is the Lord of your heart; But, when the seas frown, and the hurricane blows, With how much amaze shall he start. How happy the lovers who calmly defy The fair one they cannot esteem; But yet in the midst of your scorn let me die, Ever I live to be frigid like them.",
        "author": "The Bee, or Literary Weekly Intelligencer",
        "date": 1791    },	
{
        "text": "What youth, perfumed with liquid dews Distilled from the ambrosial rose, Shall, Pyrrha, in your grot recluse, To you his ardent flame disclose! For whom those eyes with radiance glow? For whom those golden ringlets shine? Unhappy youth! a scene of woe, Unalterable woe, is thine! As when the gentle breezes rise, And calm the wat’ry realms appear; Anon the tempest rends the skies, And fills the sailor’s heart with fear: Alike his fate– condemned to burn For you, but never to prevail; Anon his lovelorn heart must mourn The altered gods and faithless gale! How wretched those who view that face Where every feature seems divine, Where every beauty, every grace, To gild that treacherous soul combine! The temple of old ocean’s God Can witness to my faith sincere; That I those sacred walls have trode, And paid my votive offering there.",
        "author": "William Moscawen",
        "date": 1793    },	
{
        "text": "Where roses flaunt beneath some pleasant cave, Too charming Pyrrha, what enamoured boy, Whose shining locks the breathing odours lave, Woos thee, exulting in a transient joy? For whom the simple band dost thou prepare, That slightly fastens back thy golden hair? Alas! how soon shall this devoted youth Love’s tyrant sway, and thy changed eyes deplore, Indignant curse thy violated truth, And count each broken promise o’er and o’er, Who hopes to meet, unconscious of thy wiles, Ingenuous looks, and ever facile smiles! He, inexperienced mariner! shall gaze In wild amazement on the stormy deep, Recall the flattery of those sunny days, That lulled each ruder wind to calmest sleep. ’Twas then, with jocund hope, he spread the sail, In rash dependence on the faithless gale. Ah wretch! to whom untried thou seemest fair! By me, who late thy halcyon surface sung, The walls of Neptune’s fane inscribed, declare That I have dank and dropping garments hung, Devoted to the God, whose kind decree Snatched me to shore, from an overwhelming sea.",
        "author": "Anna Seward",
        "date": 1799    },	
{
        "text": "What essenced youth, on bed of blushing roses, Dissolves away within thy glowing arms? Or with soft languor on thy breast reposes, Deeply enamored of thy witching charms? For whom do now, with wantonness and care, Thy golden locks in graceful ringlets wave? What swain now listens to thy vows of air? For whom doth now thy fragrant bosom heave? Alas! how often shall he curse the hour, Who, all-confiding in thy winning wiles, With sudden darkness views the heavens low’r, And finds, too late, the treachery of thy smiles. Wretched are they, who, by thy beauty won, Believe thee not less amiable than kind: No more deluded, I thy charms disown, And give thy vows, indignant, to the wind.",
        "author": "John Davis",
        "date": 1799    },	
{
        "text": "What graceful youth, in liquid odours drest, Beneath some Caverns grateful shade, On beds of fresh-blown roses laid, Inconstant Pyrrha! with thy charms is blest? For whom thy radiant tresses dost thou twine In simple elegance arrayed? Too lovely, too bewitching maid! Ah luckless they to whom untried you shine! How little recks he of the faithless wind, Who now enjoys thy golden smile, And, fondly credulous the while, Still free, still constant, hopes thy love to find! How is he doomed to mourn, unhappy youth! And gaze with looks aghast, and weep; While blackening tempests ride the deep, His altered Gods and ill-requited truth! Thus must the lost unwary youth complain; Long since escaped the billows, I My votive tablet hung on high, And dripping weeds, in Neptune’s sacred fane.",
        "author": "Anonymous",
        "date": 1799    },	
{
        "text": "What gentle youth, in flowers and fragrance drest, Now clasps thee, Pyrrha, in his glowing arms? With touch of amorous fire unzones thy breast, And riots, licensed, o’er its heaving charms? For whom is flooded now that simple grace Which plaits thy robe in many a careless fold? For whom, with blushing radiance lights thy face, And float those wavy curls of threaded gold? Alas, for him! too soft confiding youth, Who trusts the transient summer of thy smile, Receives thy easy vows for tests of truth, Nor dreams how foully fair is women’s guile. Infatuate Dupe! too soon, yet, ah! too late Thee perjured, and, himself undone, he’ll find; Then with availless curses brand his fate, Upbraid the world, and call the Gods unkind! This heart a stoic shield of caution saves, And lets me view, uncharmed, thy Circe-form; So Ocean’s soft, clear front, and sun-lit waves The Calm invites but then I dread the Storm!",
        "author": "Gazette of the United States",
        "date": 1800    },	
{
        "text": "Rosa! in yonder pleasant cave, Where murmurs sweet the streamlet wave, What graceful youth invites thy soul, To smile on Circe’s nectar bowl? And say for whom with blushing care, You twine in braids your golden hair; Maid, in whose beauteous form we see, The spirit of Simplicity! Fond youth beware! though now you lie, Secure beneath a cloudless sky; And dream that every smile will prove, A herald of eternal love: The warning of the Eolian shell, Sounds with a more tempestuous swell; And through that sky so clear, so warm, Will rush the demon of the storm; And rouse thee from thy raptured sleep, To wonder at the change, and weep! Lured by the bright and sunny beam, That warmed the bosom of the stream; I launched my little bark from shore, I launched it– to return no more! Escaped the dark and whelming wave, My grateful vows to Heaven I gave; And dripping with the foamy brine, My garment hangs on Neptune’s shrine; To warn whom sunny skies beguile, Or Rosa’s more deceitful smile!",
        "author": "William Isaac Roberts",
        "date": 1801    },	
{
        "text": "Pyrrha, say what fine-formed boy Urges thee to am’rous joy, All on roses sweetly laid, In some grotto’s pleasant shade, Who with perfumes that so shed Liquid fragrance is overspread? Say for whom, thus plain and neat, Thou thy sunny hair dost plait? Ah, how often shall he wail Gods that change, and faith that’s frail; Wail, when he, unpracticed, finds The seas roughen with dark winds! Lapt in golden pleasures, who, Credulous, enjoys thee now; Hopes you’ll ever, ever prove Beauteous, vacant to his love; While he little seems to know What deceitful gale may blow? Wretched they, whom charms so bright Unexperienced shall delight! As for me, this wall declares, Which my votive tablet bears, That my drenched weeds hang on high To the sea’s great Deity.",
        "author": "John Nott",
        "date": 1803    },	
{
        "text": "What gentle youth, my lovely fair one, say, With sweets perfumed now courts thee to the bow’r, Where glows with lustre red the rose of May, To form thy couch in love’s enchanting hour? By zephyrs waved, why does thy loose hair sweep In simple curls around thy polished brow? The wretch that loves thee now too soon shall weep Thy faithless beauty and thy broken vow. Though soft the beams of thy delusive eyes As the smooth surface of the untroubled stream; Yet, ah! too soon the ecstatic vision flies Flies like the fairy paintings of a dream. Unhappy youth, oh, shun the warm embrace, Nor trust too much affection’s flattering smile; Dark poison lurks beneath that charming face, Those melting eyes but languish to beguile. Thank heaven, I’ve broke the sweet but galling chain, Worse than the horrors of the stormy main!",
        "author": "Thomas Chatterton",
        "date": 1803    },	
{
        "text": "What slender youth, all-odored, presses Thee, Pyrrha, in the roseate shade? Fow whom thine auburn-flowing tresses, Simply becoming, dost thou braid? How oft, alas, by thee forsaken, Shall he his altered fate deplore; View the dark deep, that storms awaken, And wonder at the unwonted roar; Who now enjoys, too fond believer, Thy golden charms; who always kind And lovely deems his dear deceiver, Forgetful of the faithless wind. Ah wretch, by whom untried thy beauty! My votive tablet on his fane Shews my dank weeds, with grateful duty Hung to the Ruler of the Main.",
        "author": "E. L. Swift",
        "date": 1803    },	
{
        "text": "With liquid fragrancy bedewed In some cave’s grateful solitude, What Youth on many a rose reclined Now woos thee, Pyrrha! coyly kind? And say, for whose unguarded eye In thy adorned simplicity, Upbind’st thou with lascivious care The loose gold of thy flowing hair? How oft shall he, alas! in vain, Thy perjured faith and Heav’n arraign; And wond’ring view the whirlwind sweep Turbid and dark the ruffling deep! Tho now too credulous he prize Thy favour’s golden witcheries; And deem thou wilt for ever prove Thus amiable and free to love: Nor heeds the gale’s fallacious rest That flattering smooths the billow’s breast. Ah! wretched who unknowing try ‘The smiling treachery’ of thine eye! Me late escaped the ocean swell The tablet-pictured wreck may tell; Suspended on the solemn fane Drop my dank vestments from the main; Votive to Him who stills the wave, Mighty and merciful to save!",
        "author": "Charles Abraham Elton",
        "date": 1804    },	
{
        "text": "What slender youth around thy charms, Perfumed with odours, twines his arms, On blushing roses loosely laid, Deep in some grotto’s grateful shade? Who bids thee bind thy auburn hair, Thou Charmer negligently fair? Alas! how soon will he deplore Thy altered kindness, his no more. The unskilful boy amazed will weep, That storms deform the changeful deep. What youth, now favored, hopes to find Thee always lovely, always kind? Deceitful fair one, he shall prove The wind less wavering, than thy love. Unhappy they, whose hearts you charm, Who know not of your power to harm. For me, escaping from the wave, Favored by Neptune strong to save, I in the temple of the god A votive tablet have bestowed, And the wet garments, which I wore, When shuddering I attained the shore.",
        "author": "Andrews Norton",
        "date": 1805    },	
{
        "text": "Pyrrha, the slender youth who courts thy love, Bathed in rich odours, on fresh roses laid, Beneath the grateful shade Of some cool cavern or embowering grove, For whom thy golden hair thou dost unbind, (Simple in elegance) though now most blest, Of thy whole heart possessed, He hopes thee always free and always kind, Alas poor wretch! how often shall he deplore Thy false love, changing with the changing skies, And stormy seas, that rise Black with rude winds, and bear him from the shore, (Too weakly trusting to the treacherous gale!) Ah hapless they on whom thy untried smile Beams only to beguile; Who see thee fair, but know not yet how frail! My votive tablet still records the hour When, rescued from the vexed and stormy wave, My dripping weeds I gave, A grateful offering to the watery Power.",
        "author": "John Herman Merivale",
        "date": 1806    },	
{
        "text": "What graceful youth, on roses soft reclined, Bedewed with liquid perfumes, Pyrrha fair! Caresses thee, beneath The cavern’s cooling shade? For whom dost thou in simple neatness bind The golden locks that wave around thy brows? Alas! how he will mourn The altered gods’ decrees! Unconscious he will wonder at the storm, Who now too credulous enjoys thy smiles, And dread the black’ning clouds That lour above his head. Alas! for him to whom untried thou seemest Splendid in charms, great Neptune’s sacred walls My dripping robes display, A votive offering.",
        "author": "J. P. C., The Poetical Magazine",
        "date": 1809    },	
{
        "text": "Impart, my fair-one! and with truth, What well-shaped beau, in bloom of youth, With roses decked, and odours sweet, Invites you to some cool retreat? Or say, thou fairest of the fair, For whom you bind your locks of hair, Or dress so elegantly plain, To bless with love some happier swain? And yet, alas! in time to come, He’ll have to mourn his wayward doom, And view with horror and surprise The storms of jealousy arise. For he who now enjoys your care, Incautious of your flattering air, Will think your love for him secure As his for you– sincere and pure. But, wretched they! to whom untried Your beauty charms in all its pride; While I, escaped your witching pow’r, Will thank kind Neptune every hour.",
        "author": "Mr. Vaugham, The Poetical Magazine",
        "date": 1810    },	
{
        "text": "O say, where all-propitious Love Hallows with shade the deep alcove, Where roses all their sweets exhale And perfume every passing gale, What graceful boy, in these gay bowers, Profusely spends his languid hours; Clasps to his heart its dearest treasure, And fondly tempts the roving pleasure? For whom you let those tresses flow Along that cheek’s delicious glow, And all the charms of dress dispense In easy, artful negligence. Unhappy youth! these moments flown, Thy bonds of love the fates will sever; And she, who now is all thy own, Destroy thy dream of bliss for ever! For, ah! thou can not see the art That lurks beneath that radiant eye And little dost thou know the heart That laughs at frequent perjury. Lais! the breast which lately beat With ardent and tumultuous heat, Has now escaped from ev’ry pain, And gained its tranquil throb again; Doomed never more to feel the sigh, That speaks the power of beauty’s eye; Nor, with unceasing pain, to prove The pangs of unrequited love. Taught to be cold, its hopes and fears Are buried in the lapse of years; And, all its former passion dead, Its cares and pleasures both are fled.",
        "author": "T. S. M., The Theatrical Register",
        "date": 1812    },	
{
        "text": "What graceful youth whose dewy ringlets breathe Their essenced sweets, Pyrrha, mid roses laid Lascivious by thy side, seeks to persuade Thy melting charms, some blissful bower beneath? Whom now to please, thy bright locks dost thou wreathe Easy in simple grace? Ah! soon to fade Thy faith and all his joys, while storms invade The blackening waves which little he foreseeth! Who trusting to thy soft and golden hour Ever unoccupied and kind as now, Hopes thee, nor doubts the fair and flattering breeze New victim of thy false smile’s magic power. For me, the pictured wall declares my vow, Wet from the wreck, no more to tempt the seas.",
        "author": "Robert Morehead",
        "date": 1813    },
{
        "text": "What airy youth, whose locks exhale The liquid odour’s balmy gale, Beside thee, Pyrrha, now reposes Within the favourite bower of roses? Tell me for whom that golden pride, Thy hair, with graceful ease is tied, And all thy vesture, flowing free, Is rich in sweet simplicity. Alas! the fondly-trusting boy Who hails thee now his heav’n of joy, Nor, all unpracticed, ever foresees The veering of the faithless breeze, But paints thee still to fancy’s view Enchanting ever– ever true; How will he start, when first he finds His ocean tossed by angry winds? How will he wonder!– how bewail His easy faith in one so frail! How often accuse the fickle Powers That turned to thorns his couch of flowers! Ah! wretched he, the fond believer Who knows thee not, thou sad deceiver! But I have ’scaped that wreck of love; And long shall grateful offerings prove The mercy of the Power that bore A struggling, sinking youth to shore.",
        "author": "E. W., The Gentleman’s Magazine",
        "date": 1813    },
{
        "text": "Pyrrha, what ardent stripling now, In one of thy embowered retreats, Would press thee to indulge his vow Amidst a world of flowers and sweets? For whom are found thy tresses bright With unconcern so exquisite? Alas, how oft shall he bewail His fickle stars and faithless gale, And stare with unaccustomed eyes, When the black winds and waters rise, Though now the sunshine hour beguiles His bark along thy golden smiles, Trusting to see thee, for his play, For ever keep smooth holiday! Poor dazzled fools, who bask beside thee! And trust because they never tried thee! For me, and for my dangers past, The grateful picture hangs at last Within the mighty Neptune’s fane, Who snatched me, dripping, from the main.",
        "author": "Leigh Hunt",
        "date": 1814    },
{
        "text": "What shapely youth, on heaps of roses laid, And bathed with dropping odours, woos thee now In gloom of pleasant grot? for whom dost thou Thy yellow locks, oh Pyrrha, backward braid So simply elegant? how oft shall he On woman’s faith, and changed Gods bewail; And view, with unaccustomed wonder pale, The winds scowl dark upon the troubled sea! Who, credulous, in thy gilded beauty blest, Now fondly deems thou shalt for ever prove Thus amiable, thus open to his love; Unweeting of the gale’s fallacious rest. Ah wretches! that with inexperienced eye Gaze that serenest brow! I, shipwrecked, flee, With painted storm, to the strong God of sea, And hang my dank weeds in his temple high.",
        "author": "Charles Abraham Elton",
        "date": 1814    },
{
        "text": "Pyrrha! what youth with odours crowned, On rosy beds shall thee caress; In shady grots with myrtle crowned, What youth wilt thou vouchsafe to bless? For whom dost thou thy golden hair In flowing tresses loosely bind, Genteelly plain, with easy air, Say unto whom wilt thou be kind. Alas! poor youth, he’ll often bewail, The plighted vow that’s broke by you, His prayers to Cupid wont prevail, Venus is deaf to all his woe. A stranger to thy faithless mind, He thinks he is for ever blest, Deceived, forgot, he’ll shortly find, The waves are ruffled in thy breast. Unhappy those whom you detain, Enamoured with your faultless shape, My drooping clothes in Neptune’s fane Declare to all my hard escape.",
        "author": "Anonymous, The New Monthly Magazine",
        "date": 1814    },
{
        "text": "What slender youth, with liquid fragrance dewed, Wooes the now, Pyrrha! in some pleasant grot, Throned in some pleasant bow’r, with roses strewed? For whom thine amber tresses does thou knot, Simple in Elegance? How oft Shall he, who hopes thee still thus soft, Still lovely, thus in golden temper gay, Of the false gods, and his fond faith complain, As, unsuspicious of the changeful day, Aghast he sees black whirlwind sweep the main? Ah! hapless they, to whom, yet new, Thou shinest without a cloud in view! The votive tablet and the dripping vest, Hung to the Monarch of the Sea, Proclaim me landed from wild Ocean’s breast, Escaped from wrecking storm and thee.",
        "author": "S., The Morning Herald",
        "date": 1817    },
{
        "text": "What fair youth, among roses lying, Courts thee, Pyrrha, to the grove, Where fresh odours round him sighing, Seem to whisper– sweet is Love? Grace is in thy neatness smiling, Golden tresses round thee flow; Say for whom, the breeze beguiling, Bids thou them in wreaths to glow? Oft on faith, and Jove’s displeasure, Shall the hapless youth complain, And dark winds in horrid pleasure See to mountains swell the main! Thoughtless who is now enjoying Thee in sunny charms untold; Who, nor heeds the storm destroying, Hopes thy constant smile of gold? They shall sink in deepest sorrow, Who, unknowing all the wiles Thy too-treacherous heart can borrow, Glory in thy transient smiles! Quick, for so the vow was framed, On the holy wall shall be Dank and dropping weeds unnamed, Offerings to the god of sea.",
        "author": "Anonymous, The Gentleman’s Magazine",
        "date": 1817    },
{
        "text": "Who, in the grotto’s pleasing gloom, Extended ’neath the rosy bower, Breathing Arabia’s soft perfume, Trifles with thee the happy hour? For whom, in artless beauties drest, Do those soft ringlets shade thy face? For whom enrobes the flowing vest, That form so fair, in simple grace? How oft, unused the storm to bear, Shall he thy fickle faith deplore, When angry tempests cloud the air, And swept by winds the billows roar? Who now enjoys thy smiles alone, (Ah! ignorant of the changeful wind,) Who fondly deems thee all his own, Thus ever pleasing, ever kind? The form in votive urn expressed, Tells, that escaped the raging wave, Suspended high my dripping rest, To Ocean’s powerful God I gave.",
        "author": "Anonymous, The Repository of Arts",
        "date": 1817    },
{
        "text": "What graceful boy in rosy bowers, Bathed in sweets of dewy flowers, Circles thee, Pyrrha, in his arms? Maid of the amber hair, and snowy charms. Finely formed and simply clad, Alas, amazed and wildly sad, He shall see dark storms arise, Ruffled seas and lowering skies. Securely now entranced in blisses, Brief as thy insidious kisses, Swift as his image from thine eyes, Fleet all his amorous revelries. Wretches, who thoughtless embark with thee, Prophetic learn your fate of me, Where tablets on yon shrine display, My vests suffused with the foamy spray.",
        "author": "Anonymous, The Gentleman’s Magazine",
        "date": 1817    },
{
        "text": "What slender youth, all essenced over, In sweet alcove or rosy bower Now woos thee, Pyrrha, to be kind? For whom those tresses dost thou bind, Thus simply neat? O how shall he, Poor youth! bewail the boisterous sea, Rough with black tempests! How accuse Capricious Gods, and broken vows! Fond dupe! he hopes– so sweet that kiss! Thou’lt still be witching, still be his. What treacherous gales beset his way, Ah! little knows he! Hapless they, Who ne’er thy faithless smiles have tried!– That I have escaped the whelming tide, A tablet and my dripping vest, Hung up in Neptune’s fane, attest.",
        "author": "Francis Wrangham",
        "date": 1821    },
{
        "text": "What slender youth, O Pyrrha fair! Courts thee ’mid roses sweet? For whom binds thou thy yellow hair So elegantly neat? Alas, alas, how often shall he Of fickle gods complain, And, unaccustomed, wond’ring see The ragings of the main. Who now enjoys thy faithless smiles And thee all-lovely sees; Nor knows thy false alluring wiles, Nor fears the treacherous breeze. Unhappy they, to whom untried Thou seemed so passing fair; Sweet Hope now sparkles o’er the tide, Soon followed by Despair. Like others I, with keen delight, Beheld thy witching smiles; Thou seemed so lovely in my sight, I thought not on thy wiles. When whelmed beneath the foaming tide, When hope of life seemed vain, On Neptune’s mighty power I cried To save me from the main. My vestments dripping from the wave To Ocean’s god I give, And on the votive stone engrave, ‘By Neptune’s hand I live.",
        "author": "Mary Bailey",
        "date": 1822    },
{
        "text": "What youth genteel, bedewed with sweets, In bowers delightful Pyrrha greets, Where roses shed perfume! For whom you braid your auburn hair, And ev’ry blandishment prepare, To best ensure his doom! Alas! how oft thy faithless love And changeful pow’rs he’ll surely prove, And treacherous seas declaim, Who, credulous, with ardent joys Your constancy, he thinks, employs, And inexperienced flame. Like him who trusts the Ocean’s calm, And thinks th’ inconstant winds to charm, And on the waves repose; My shipwreck lately undergone, My garments drenched in tempests, own, Sacred to Neptune’s Laws!",
        "author": "James Usher",
        "date": 1824    },
{
        "text": "Whose the vows that breathe so tender Through yon grotto’s clustering roses? Whose the form so soft and slender, That on Pyrrha’s breast reposes? Ha! why with such neatness now Shade those glossy locks her brow? He who now enjoys thy smiles, Thoughtless mortal, soon shall rue, Soon shall mourn thy luring wiles, Marvelling that he thought thee true, As the waves, which calmly slept, By the blackening storms are swept. Though he think thy bosom glows With a passion pure as lasting, Ah! too soon a tempest blows, All his dreams of rapture blasting. Woe’s me for that hapless wight Knowing but thy beauty bright. Now to shew the deadly spell, High within the sacred fane Shall the votive tablet tell All my perils on the main, And that, escaped, I grateful bring My sea-drenched robes to ocean’s King.",
        "author": "Izaak Marlowe",
        "date": 1824    },
{
        "text": "Come tell me, Pyrrha, what fair boy Perfumed with sweets of gentlest power, Was pleading Love’s delicious joy, Within yon favouring rosy bower? O! say for whom, seductive girl, Your golden locks, you gently curl, Nymph, whose sweet face and form might be A temple for simplicity! Alas! how oft shall he deplore False vows, and oaths that bind no more; And witness, with unknown emotion, The storms of Love’s deceitful ocean Who, won with bliss, so fond, so new, Hears thy false vows, and thinks them true, Whom, now, thy potent charms allure, And thinks such frailty can endure. I, too, in love’s young, ardent dream, Launched my frail bark upon the stream, But soon before the tempest driven, I gave my grateful vows to Heaven. And lo! on Neptune’s temple wall, My garments, dripping with the sea, Proclaim how I escaped the thrall Of frail inconstancy and thee!",
        "author": "George Fleming Richardson",
        "date": 1825    },
{
        "text": "What lovely youth in roseate bower Shares with thee the present hour, Shedding perfume round? For whom, fair Pyrrha, lovely maid, Are all thy native charms displayed, And golden tresses bound? Alas! poor youth, too soon he’ll find Thy faithless vows and fickle mind, And mourn his altered fate! Too soon the winds in boisterous measure, Will ruffle o’er his sea of pleasure And change his happy state; Though now he fondly thinks those charms Can never bless another’s arms, Nor sees the lurking storm! What pangs are they not doomed to prove, Who knowing not thy changeful love Gaze on that heav’nly form! For me– too happy to escape with life, When, mid the waves and tempests’ strife, I’ve struggled through the flood, In Neptune’s sacred Dome I set My garments, pictured dripping wet, To mark my gratitude.",
        "author": "Anonymous, The Parthenon",
        "date": 1825    },
{
        "text": "For whom in undulating tresses Does Pyrrha wreath her golden hair? What slender youth, who lightly presses A bed of roses, courts thee there? Bedewed with an ambrosial river, The credulous lover little dreams The bark of pleasure storms shall shiver, Now dancing o’er those fragrant streams. Alas! how oft of changeful heaven And broken faith, shall he complain, Who now, to thy delusions given, Enjoys a dream so bright, so vain! When he beholds the face of ocean Roughened with blackening winds, and views The storms unwonted, whose commotion His paradise in ruin strews! Unhappy all to thee a stranger Who think that thou art fair! for me, Escaped from their impending danger, The stern controller of the sea Beholds upon his shrine suspended My garments, dripping from the main, The wreck, whose perils now are ended, To call to memory again.",
        "author": "James Nack",
        "date": 1827    },
{
        "text": "What tender stripling, on luxuriant roses, Now clasps thee Pyrrha? Laved with soft perfume, What dainty boy upon thy breast reposes Beneath the twilight of thy grotto’s gloom? Still, delicately neat, thy finger closes In wreathes thy sunny hair– but ah! for whom? Alas! how oft, in agony, will he Faith and the changed gods lament: how oft, In strange astonishment, behold the sea Rough with black storms, in billows hurled aloft; Who now, too fondly, basking there may be, On its bright bosom, beautiful and soft! Who, ever free and constant, ever kind, Expects to see thee, since he sees thee now: To treacherous tempests credulously blind, Aye brooding ’neath the sunlight of thy brow. Oh, wretched they, who hope in thee to find, The calm of love, and love’s unbroken vow!",
        "author": "W.P. Smith",
        "date": 1827   },
{
        "text": "Pyrrha! what slender stripling now, In some retired and sweet alcove, While liquid odours round him glow, Entreats thee to indulge his love? For whom binds thou in wreaths thy golden hair, With unobtrusive elegance and care? Alas! how oft shall he complain Of altered faith and deities, And look upon the ruffled main And blackened sky with strange surprise, Who now enjoys thy smiles, and hopes thou will be From change and cruelty for ever free! Ah, hapless they whom thou, untried, Allures with fatal loveliness! For my safe rescue from the tide I have hung up my dripping dress, As on the wall my votive picture tells, To Him, through whom the ocean sinks or swells.",
        "author": "James Stringer",
        "date": 1829   },
{
        "text": "How can thou look so good and fair, Maid of the perjured vow, For whom will you braid your golden hair In tresses o’er that placid brow? Thou whispered love one fatal hour Which stole away my senses quite; But ah! that love was like the flower, That blooms at morn and dies at night. Unhappy they whose hearts you win, Who see thee decked in beauty’s pride, And think that all is pure within. Since thou art angel-like outside. Away! thy beauty’s dead to me, Resistless once– the charm is over, I well escaped thy treachery Away! I’ll see thy face no more.",
        "author": "Anonymous, Kerry Evening Post",
        "date": 1831  },
{
        "text": "What slender youth on bed of roses, Pyrrha, by the side reposes, With odours perfumed sweet, In shady grot reclined? And when her waving auburn tresses Plain in her neatness Pyrrha dresses, O whom is it to meet? For whom art thou so kind? Alas, how oft will that fond boy, Who now so blindly can enjoy Thy venal beauties, weep Thy broken vows of love, When all thy perjury he finds? And, wond’ring at the rough’ning winds That brush the darkling deep, Will woman’s anger prove! He hopes, unconscious of thy wiles, To bask for ever in thy smiles, And have thee his alone. Yet more are those unblest Who all untried thy charms admire; In token, then, of my desire Before great Neptune’s throne I hang my dripping vest.",
        "author": "Martin F. Tupper",
        "date": 1833   },
{
        "text": "On many a rose reclined, All odour-bathed, in some cave’s grateful lair, Say for what slender youth shall Pyrrha bind, Simple in simplest garb, her amber hair? How of the broken vow, And the changed Gods, the credulous dupe shall weep, As o’er thy mind’s smooth calm, so halcyon now, Wond’ring he views the dark winds, roughening, sweep! He now thy golden charms Enjoys, and hopes thee aye as prompt to please; Nor dreams, unconscious lapped within thine arms, Of all that waits him in the faithless breeze. Ah! hapless they for whom Thou glows– untried, as yet, thy treacherous love; But I am one escaped the watery tomb, And hang my robes the Sea-God’s shrine above!",
        "author": "Anonymous, The New Monthly Magazine",
        "date": 1833   },
{
        "text": "What slender youth whom liquid odors late, Courts thee on roses in some pleasant cave Pyrrha?– for whom with care Binds thou thy yellow hair Plain in thy neatness? Often alas! shall he On faith and changed Gods complain, and sea Rough with black tempests ire Unwonted shall admire! Who now enjoys thee credulous– all gold For him still vacant, lovely to behold Hopes thee: of treacherous breeze Unmindful. Hapless these To whom untried thou seemest dazzling fair. Me Neptune’s walls, with tablet vowed, declare My shipwrecked weeds unwrung To the sea’s potent God to have hung.",
        "author": "Anonymous, The Southern Literary Messenger",
        "date": 1833   },
{
        "text": "Pyrrha, who now, mayhap, Pours on thy perfumed lap, With rosy wreath, fair youth, his fond addresses? Beneath thy charming grot, For whom, in gay love-knot, Playfully dost thou bind thy yellow tresses? So simple in thy neatness! Alas! that so much sweetness Should ever prove the prelude of deception! Must he bewail too late His sadly altered fate, Chilled by a bleak tempestuous reception, Who now, to fondness prone, Deeming thee all his own, Revels in a long dream of future favour; So bright thy beauty glows, Still fascinating those Who have not learnt how apt thou art to waver. I the false light forswear, A shipwrecked mariner, Who hangs the painted story of his suffering Aloft over Neptune’s shrine; There shall I hang up mine, And of my dripping robes the votive offering!",
        "author": "Francis Sylvester Mahony",
        "date": 1836   },
{
        "text": "Pyrrha! what graceful boy Woos thee in bower of joy Steeped in moist fragrance, amidst many roses? For whom, O witching fair! Braids thou thy yellow hair, Simple in elegance, as he reposes? Alas! how oft the fleetness Of thine enchanting sweetness, Thy different aspect, thy most brief faith-keeping Like Ocean dark with gales That rend the unreefed sails Must that poor Youth, unused to thee, be weeping! Ah, credulous! he deems, Thou art all gold, and dreams, Thou shalt be ever listless, ever smiling; Nor thinks the fickle breeze Will change, and lash the seas To tempest, those who know thee not, beguiling. Hapless that dazzled crowd! For me, my picture vowed, High on the sacred wall sets forth my story; How I, from storms to rest Escaped, have hung my vest All dripping, to the potent Sea-God’s glory.",
        "author": "W. H. Budden",
        "date": 1836   },
{
        "text": "What graceful youth perfumed with liquid sweets, On roses couched, enjoys thy soft caresses? Within a pleasing grot, thee, Pyrrha! greets For whom does thou adorn thy golden tresses In simple elegance? alas how oft He’ll mourn his altered gods, thy faithless soul, With gaze unwonted see black storms aloft, And seas which darkly with the tempest roll. Fond fool! he now enjoys thee bright and gay, And hopes to find thee his, and lovely still; Ah! faithless breeze of hope; ah! wretched they, Who in thy beauty will not dream of ill. A votive tablet on a sacred shrine Will shew that I, escaped a dangerous sea, Have hung my garments moist with ocean brine An offering mighty sea-god unto thee.",
        "author": "H. Halloran",
        "date": 1836   },
{
        "text": "Pyrrha, what youth, with liquid odours sweet, Courts thee on roses in that cool retreat? O say, for whom thou bindest thy golden hair, More lovely plain, than decked with jewels rare. Alas! how oft will he in tears bemoan The gods no longer kind, thy vows foregone; And, unaccustomed to the stormy deep, Affrighted quail when winds its bosom sweep. He, who enjoys thee now all smiles and love, And hopes, with fond simplicity, thou’lt prove For ever amiable, for ever free, As well might wish the gale should changeless be. Oh wretched they, to whom thou shinest untried Like placid ocean’s sun-lit dancing tide! Yon votive tablet in the sacred fane, And those my vestments dripping from the main, Are grateful tributes to the potent god Who sways the sea-storm in its fiercest mood.",
        "author": "H. L. Clark",
        "date": 1837   },
{
        "text": "What perfumed boy, with rosy wreath, Now blandishes with thee beneath Some pleasant bow’r? For whom, my Pyrrha, do you dress Your hair, in winning simpleness? How soon the hour When he shall mourn o’er thy deceit, And rail at his how altered fate; And wond’ring prove The waters rough with black’ning storm, Who now enjoys thy golden form In dreaming love; Who blindly, fondly, hopes to find Thee ever constant, ever kind; Nor fears, nor knows the treacherous wind Thrice wretched he Who loves, but knows thee not! For me The votive tablet will attest I’ve offered up my dripping vest To Neptune, God of the sea.",
        "author": "Robert James",
        "date": 1838   },
{
        "text": "Tell me Pyrrha, who is he That, with scented locks, In thy rose bower kisses thee Neath the shady rocks? For whom is bound thy golden hair Sweetly wreathing, void of care? Oft, the Gods he would adore Vows unkept by thee; Oft, the Gods he would adore Frowning, he shall see; Oft, astonished, see the main All afoam with wind and rain, Who believes thou’lt constant prove, With thy beauty blind; Heedless, while he lives in love, Of the faithless wind! Ah how wretched, all on whom Unaware, thy beauties bloom! As for me, experienced well, Rescued from the main, And mindful of the tempest’s swell, I’ll hang in Neptune’s fane A picture of that stormy sea, And garments drenched in ocean spray.",
        "author": "Patrick Branwell Bronte",
        "date": 1840   },
{
        "text": "What slender youth, whom many roses crown, Whose hair rich liquid unguents steal adown, Wooes thee, coy Pyrrha, in some pleasant grot? For whom dost thou thy golden tresses knot Neat in thine elegance? How oft he’ll weep Thy faith and gods as mutable! The deep How oft, poor simple novice, he’ll admire Blackening beneath the savage tempest’s ire, Who now enjoys thee in thy golden days, Unconscious how the changing wind betrays; Ah, credulous! and fondly hopes to find Thee his for ever, and for ever kind. Woe unto whom thou glitterest untried! My votive picture, in his temple, tells I’ve hung my garments, reeking from the tide, Before the God, whose power the ocean quells.",
        "author": "Lawrence Peel",
        "date": 1841   },
{
        "text": "What youth genteel, bedewed with sweets In roseate-bowers, blythe Pyrrha greets, For whom your auburn locks you bind With simplest grace, love’s toils designed? Alas! how oft shall broken vows, And changeful powers, his heart oppose; How shall he gaze on raging seas, The placid stream who sails at ease, And hopes kind constancy to prove, No storm to cross the course of love! Devoted youth! Thy woe’s to come. I have already met my doom. In Neptune’s fane, a votive-plate Shews whence my dropping robes I date.",
        "author": "James Usher",
        "date": 1842   },
{
        "text": "What youth genteel, whom odours dew, Delightful Pyrrha, sports with you, Where roses breathe perfume? With simplest grace your auburn hair, Each toil for passion fond, prepare To best ensure his doom. Alas! How oft perfidious love, And changeful powers, he’ll surely prove, And treacherous seas declaim: Who, credulous, with ardent joys, Your constancy, he thinks, employs, Fond love’s ingenuous flame! Like him who trusts the ocean’s calm, And thinks th’ inconstant winds to qualm, And on the waves repose; My shipwreck, undergone of late, My dropping vest, my votive-plate, Dread Neptune’s walls disclose.",
        "author": "James Usher",
        "date": 1842   },
{
        "text": "What graceful youth, with liquid perfume glowing On beds of roses in some charming grot, Clings, Pyrrha, on the honey of thy kisses? For whom with winning heart those tresses flowing In auburn clusters binds thou in a knot? Alas! how often he’ll ponder over past blisses And fickle Cupids; and with simple heart Gaze on the swellings of the once calm sea! He– who now basks beneath thy sunny smiles, And hopes thee aye to be what now thou art, A lovely child of nature, guileless, free: He knows not of thy soft bewitching wiles! Unhappy they who love and know thee not. I, when with shipwrecked bark I escaped the main, (So tells yon sacred tablet,) humbly brought My votive garment to the Neptunian fane.",
        "author": "Barney Brallaghan",
        "date": 1842   },
{
        "text": "What dainty youth with liquid perfume sweet Bedewed, caresses you, O Pyrrha! hid In some dark grotto, prostrate at your feet, Profusion of the odorous roses ’mid? For whom do you now fillet up your hair That golden hair so delicate and fine? Alas! how often will that youth despair, And at your perfidy and falsehood pine. Deploring the estranged gods, and trough His inexperience be amazed with seas Which rough and blackening storms will quickly brew, Who, credulous, now enjoys the precious breeze; And ignorant of the faithless gale, believes That you will ever unto him be true. Wretched is he who thus himself deceives. The sacred wall of Neptune’s fane doth show By votive tablet, consecrated there Unto the potent god wet garments are.",
        "author": "R. B., Lloyd’s Companion",
        "date": 1842   },
{
        "text": "Pyrrha, say why with such care Braidest thou thy yellow hair; In some grotto’s cool retreat, Sheltered from the sun’s fierce heat, What trim-waisted, perfumed youth, Now believes thy faith and truth? Ah! how often will he mourn The gods averse, thyself forsworn! He who loves but thee alone, Fondly fancies thee his own; Soon will he his idol find False as waves or changing wind. Hapless they, on whom, from far, Thou dost shine– a guiding star; Offered on great Neptune’s shrine, Ingratitude’s no sin of mine; Me my dripping garments shew, escaped from such a sea of woe.",
        "author": "Thomas D’Oyly",
        "date": 1843   },
{
        "text": "What graceful youth, where roses bloom,– Besprinkled o’er with rich perfume Engages thee in pleasant grot? Say, Pyrrha, say for whom the knot To which those golden locks you braid, In nature’s simple loveliness arrayed? Alas! how oft will he deplore Faith, and the gods– the same no more! Whose inexperience– wondering– eyes The stormy waves, and blackened skies; Who clasps thee– credulous to find A treasure ever free and ever kind, Nor dreams of the inconstant breeze. Ah, hapless! whom– untried– you please! The sacred wall– the sea-god’s care Shows, by a votive tablet there, My dripping vest suspended high– An offering to the ocean-deity!",
        "author": "John Scriven",
        "date": 1843   },
{
        "text": "What slender youth, with sweets besprent, And crowned with rosy flowers, Pyrrha, sues thy coy consent Within the pleasant bowers? For whom dost thou with simple art Bind up thy yellow hair? Ah! oft will he deplore the heart He trusted to thy care, And the changed gods, but late so kind; And stand amazed to see, Poor novice, how the cloudy wind Stirs up the bristling sea! Who now, too much believing boy, Enjoys thy golden charms, Expects a heart without alloy, And ever open arms; Expects,– nor knows the treacherous air; Oh, hapless men are they, On whom, an unattempted fair, Thou shinest to betray. The sacred wall can show for me, By votive slab expressed, How I to the saving God of Sea Have hung my dripping vest.",
        "author": "William Caldwell Roscoe",
        "date": 1844   },
{
        "text": "With liquid odours drippling, What all-impassioned stripling, In grotto cool, ’mid many a rose, His arms around thee, Pyrrha, throws? For whom, in simplest, neatest fold, Are backward bound those locks of gold? Alas! how oft shall grieve him now, The Gods estranged, thy broken vow; How him unwonted shall surprise, On temper’s wave as black winds rise, Who reaps thy golden charms, secure Thy love for him will aye endure! Vacant for him thy heart, nor sees How soon will shift thy favour’s breeze; How wretched they by whom thy tide Of fickle love shines yet untried. Me the votive tablet shows Safe from drowning; dripping clothes On the wall to Neptune hang, Type of rescued shipwreck’s pang.",
        "author": "Henry John Urqhart",
        "date": 1845   },
{
        "text": "What slender youth, perfumed with liquid odours, Woos thee, oh Pyrrha, beneath some grateful bower, Where roses languish round? For whom dost thou, With simple neatness, bind thine auburn hair? Alas! how often will he mourn thy falseness And rail against the Gods! how soon behold, With ignorant surprise, the placid lake Changed to an angry sea by scowling winds! Now he enjoys thee, golden girl, too credulous, And, all unmindful of the fickle breeze, (That is thine emblem,) deems that you will be Forever free for him, forever loving. As for myself, the picture on the wall Of Neptune’s temple, shows where Horace paid His holiest thanks that his frail barque was rescued From this enticing mermaid of life’s ocean.",
        "author": "Arthur Simkins",
        "date": 1845   },
{
        "text": "Who is that youth so well perfumed– so slender, With roses crowned, so pressing and so tender, In yonder pleasant spot, Pyrrha, for whom thy hair in golden knot Thou bindest thus with artful artlessness? Alas! how oft he’ll weep thy faithlessness, And Fortune’s lowering brow, And wonder at the tide’s waked wrath, who now Thy golden calm enjoys, and hopes to find His mistress ever constant, loving, kind! As yet he little knows How changeful are the skies: ah! wo to those Who trust thy beauty; I– the sacred wall Where hangs my votive record shows to all That to great Neptune’s shrine I offered up my garments wet with brine.",
        "author": "Eugene Lies",
        "date": 1846   },
{
        "text": "Pyrrha, what slender youth, bedewed With liquid odours, courts thee now, In yonder pleasant grotto, strewed With many a rose? For whom dost thou In braids thine amber tresses rein, So elegant, yet simply plain? How oft, alas! thy perfidy, And the changed Gods, will he deplore, And stand amazed, unused to see The waves by tempests roughened o’er, Who, fondly trusting to thy vow, Enjoyeth thee, all golden now; Who hopes thee ever his alone, Thee ever amiable to find! Alas! how little has he known The varying of the fickle wind! How hapless is the lot they share To whom untried thou seems so fair! Against the sacred wall on high My votive tablet duly set, Proclaims to all that even I Have erst my vestments dank and wet Suspended to the deity Who rules omnipotent the sea.",
        "author": "Henry George Robinson",
        "date": 1846   },
{
        "text": "Prithee! what youth, with posy and perfume Pressing his suit, apart from all the room, Has now thy ear– Oh say for whom dost braid Thy bright brown tresses, soft, seductive maid? Plain, but with all the witchery and grace That loveliness can lend to form or face, Alas! how he his hapless stars shall rue Who sees thee glisten and believes thee true, Who thinks those smiles will last, not dreaming he How that bland air is fraught with treachery. How shall he view, unused, with wondering eyes O’er the smooth surface the black billows rise! While passing suddenly from fair to foul, The Heaven of mildness gathers to a scowl. Poor innocents! ah founder sure they must Who without trial take thee upon trust. Me just escaped, and such dark perils o’er, Thy sheeny softness can engulf no more; Riding, thank Heaven, with some remains of sense, Safe in the harbour of indifference.",
        "author": "Anonymous, Hereford Journal",
        "date": 1846   },
{
        "text": "O, Pyrrha, say, what slender boy Of those whose locks sweet odours lave, Embraces thee so fair and coy, Amid the roses of a cave? For whom hindest thou thy yellow hair Flowing and free from jewels rare? Alas! how often will he weep Thy faithless love, thy broken gage And trembling gaze upon the deep Where waves roll high and tempests rage! What simple youth enjoys thy charms And clasps thee willing in his arms Who, ignorant of the changeful wind That lashes the inconstant sea, With fond reliance hopes to find A heart forever true in thee? Unhappy he whose eyes behold Thy charming face, thy locks of gold. Behold upon the sacred wall My garments dripping from the wave, A votive gift they speak to all, Of safety from a watery grave, Yet more than from the cruel sea They tell of my escape from thee!",
        "author": "Anonymous, Southern Literary Messenger",
        "date": 1848   },
{
        "text": "Say, Pyrrha, say, what slender boy, With locks all dropping balm, on roses laid, Doth now with thee in pleasant grotto toy? For whom dost thou thine amber tresses braid, Arrayed with simple elegance? Alas! alas! How oft shall he deplore The altered gods, and thy perfidious glance, And, new to danger, shrink, when sea-waves roar Chafed by the surly winds, who now Enjoys thee, all golden as thou art; And hopes, fond fool! through every change, that thou Wilt welcome him as fondly to thy heart, Nor doth not know, how shift the while The fairest gales beneath the sunniest skies! Unhappy he, who, weeting not thy guile, Basks in the sunshine of thy flattering eyes! My votive tablet, duly set Against the temple’s wall, doth witness keep, That I, whilere, my vestments dank and wet Hung at the shrine of Him that rules the deep.",
        "author": "Theodore Martin",
        "date": 1849   },
{
        "text": "What youth now courts, with sweets bedewed, The World so seeming fair, Where in some cave with roses strewed She binds her golden hair, Now plain and guileless? he full soon The fatal change shall weep, And see aghast the coming on Of black and stormy deep. All gold, fond youth, he hopes her now, And thinks that never will fail The heart at ease and lovely brow; Nor knows the treacherous gale. Ah, hapless they who deem her fair! Saved from a watery grave, I vow me in His house of prayer To Him that walked the wave.",
        "author": "Isaac Williams",
        "date": 1849   },
{
        "text": "What slender youth, that’s stolen the rose’s bloom In odours steeped, urges with thee his prayer In some cool grot, Pyrrha? for whom Dost braid thy golden hair In simple taste? Alas! each broken vow, Each blighted hope too oft shall he deplore, Amazed that seas so tranquil now Can ever chafe and roar. Heedless he revels in his golden dream, Believes thee ever loving, ever true; One fickle breeze he cannot deem May all his hopes undo. Alas for those unwarned! thou dazzlest all; My votive tablet proves the wreck I’ve braved, My dripping garments on the wall Thank the sea-god who saved.",
        "author": "G. J. Whyte Melville",
        "date": 1850   },
{
        "text": "What stripling slim, on beds of roses, Bathed in liquid odours, wooes thee, Pyrrha, in some delicious grot? For whose pleasure art thou binding Back thy flowing locks of gold, Artless in graceful niceties? Woe! how oft shall he be wailing Thy honour, and his altered gods! And on seas with murky tempests Rough, shall marvel to his fill, All unused to such a sight. He who now too fondly trusting, Enjoys thee in thy golden hour; Who still disengaged, still lovely, Hopes to find thee, recking nought Of the treacherous breeze. O hapless They, to whom untried thou shinest! As for me, with votive tablet, The hallowed wall doth show that I Have my dripping garb suspended Unto the god who rules the sea.",
        "author": "William Sewell",
        "date": 1850   },
{
        "text": "What slender youth store roses wreath, Bathed in liquid odours presses, Pyrrha, thee the sweet grot beneath? For whom binds thy golden tresses, Simple in thine adornings? How Faith and changed gods oft weep will he, And seas with black winds roughening so, Unused in blank amazement see, Who credulous enjoys thee now, Golden: who hopes thee always free, And always kind, nor aught doth know Of the treacherous gale! Wretched they To whom thou dost unproved shine! The sacred wall doth speak for me, I’ve hung those dripping garments mine, Vowed to the potent god of sea.",
        "author": "William George Thomas Barter",
        "date": 1850   },
{
        "text": "Oh! Pyrrha, nymph of pleasant caves Reclining on a couch of roses, What youth, bedewed in spicy waves, Close by thy wanton side reposes, Doth some lover, steeped in wine, Mid thy golden tresses twine Wreath of dowers, with rosy fingers, While his lip mid nectar lingers? He knows thee not, thou heartless one, Inconstant as the changing sea, But fondly hopes the smiling sun Will ever mark thy constancy. Wretched they who know thee not Pity on their hopeless lot Wo to him who first hath met Thee, the heartless and coquette.",
        "author": "J. A. Turner, Southern Literary Messenger",
        "date": 1850   },
{
        "text": "What slender youth, with softest odours laved, Seeks thee, on roses, in pleasant cave’s retreat? For whom, Pyrrha! say, thy golden tresses waved, Now bindest thou, in simple order neat? Ah! say, how oft shall he lament with tears, False faith, and varying gods, and strange, shall see Seas, rough with darkest winds, and, with sad fears, Shall wonder, beauteous, who now trusts in thee? Enraptured with thy charms, and fondly still, Hopes that thy loveliness and truth shall last, Ever unchanging, as the constant rill; Nor knows he yet thy wavering feelings past. Unhappy they, to whom, untried, unknown, Thou brightest seems. On sacred wall for me, Hang dedicate, with votive tablet shewn, My vestments damp, to potent god of sea.",
        "author": "Anonymous, The Gamester",
        "date": 1850   },
{
        "text": "Pyrrha, in thy pleasant bower, Where the clustering roses flower, Say, what youth with rich perfumes Seeks thy grot amid the blooms. Say, for whom, with simple air, Bind you thus your golden hair, Oh! how often shall the tear Fall for changing fortune here? He who too confiding lies In the sunshine of thine eyes, Soon must wail, when over the wave Dark the angry stormwinds rave. Hapless if he dreams thou’lt be Still so lovely, still as free, Inexperience deems thee fair, Yet deceptive gales are there. I my dripping vestments place Where the votive tablets grace Neptune’s fane,– as glad to be From Pyrrha as from shipwreck free.",
        "author": "George Beddow",
        "date": 1851   },
{
        "text": "What dainty youth bedewed with perfumes, Pyrrha enjoys thy favoring smile? Reclining on a bed of roses He fondly deems thee his the while. For whom fair maid are those fair tresses Bound by thy hand with special care, ‘Thy homely neatness’ courts caresses: Enraptured youth beware, beware! How soon, alas! with altered fortune Shall he thy perfidy deplore, When round him blackening storms shall gather, Where all was peace and calm before. He blindly deems thee his for ever Hopes thou wilt still be always kind; Experience, with its iron fetter, Shall teach new wonder to his mind. Unhappy those who, charmed by beauty, Fall victims to thy treacherous love Escaped, I’ll raise a votive tablet A tribute to the gods above.",
        "author": "J. A. C. B., Newry Telegraph",
        "date": 1852   },
{
        "text": "Who’s the stripling slim with liquid scents Drenched, on plenteous rose, that sues thee hard In pleasant grot? for whom Tiest thou, Pyrrha, thy auburn hair Simple in grace? How oft, alas! will he Faith and changed gods lament, and soon In strange surprise behold Black winds sweep on a ruffled sea! Now he joys to eye thee golden-bright, Hopes thee alway vacant, alway kind; Fond fool! of shifting breeze Thoughtless. Woe for the hearts, to which New thou glitters. Me the sacred wall Shows on votive board, when high I hung My dripping weeds;– a gift Gladly paid to the Seagod’s might.",
        "author": "Francis William Newman",
        "date": 1853   },
{
        "text": "What slender youth, amidst roses soft, All scent suffused, O Pyrrha fair, In pleasant cave woos thee? with care For whom binds thou thy yellow hair, So simply neat? Of faith, how oft, And changeful gods, will he complain, And shall, rough with black winds, the main His wonder, unaccustomed, gain, Who now charmed, trusts thee, golden all; Who hopes thee ever free and kind, Unknowing of the treacherous wind! Unhappy those thy splendours blind, As yet untried! The sacred wall Declares, in votive picture, me To have my dripping robes to thee Hung up, O potent god of sea.",
        "author": "J. M. L.",
        "date": 1855   },
{
        "text": "What graceful boy, with rose-crowned brow All sprinkled o’er with fragrant dews, Pressing his suit to Pyrrha now, In shadowy grot his love renews With eager vow? With simple charm of braided hair, For whom does Pyrrha weave a lure? Poor Heart! that hopes thee true as fair; Poor Heart! that in thy love secure Forgets all care. Too soon, for smiles shall be all tears, And prayers unheard, and hopes all wasted: The tempest, with fast-gathering fears, To dash the joy-cup down ere tasted, Already nears. Ah! hapless they, for whom untried Thy wanton, mocking graces shine! A ship-wrecked sailor from the tide, I’ve sought the Saviour-Sea-God’s shrine; Its walls beside, My wave-drenched garments have I flung; And gladder for the sorrow past, The votive tablets all among, To the fierce Ruler of the blast, My offering hung.",
        "author": "Anonymous, The Knickerbocker",
        "date": 1855   },
{
        "text": "What tender youth, with perfumed hair, On couch of roses thee caresses, In pleasant grotto, Pyrrha fair? For whom thou binds thy yellow tresses, With simple neatness. Ah! how oft False faith and fickle Gods he’ll weep And wind-lashed billows, tossed aloft, Will marvel at– deceitful deep! Who now enjoys thee beauteous there, Who hopes to find thee always free Unlessoned in the treacherous air And always lovely? Hapless he For whom, unknown, you shine! For me, The sacred tablet notes that I Have hung my dripping garments high, Votive to him who rules the sea.",
        "author": "Michael Joseph Barry",
        "date": 1856   },
{
        "text": "What youth, slender of form, decked with full many a rose, And bathed with a sweet stream, doth his fond suit propose, Pyrrha, under some cool grot? For whom locks of pure gold dost knot, Simply, yet with such grace? How shall he mourn, ah me! Thy faith broken, and mourn each averse Deity, And seas, rough with black storms’ ire, Unused, how shall he oft admire, Who, enjoying thee once, lured by thy golden sheen, Hoped that aye free to love and to be loved thou had been: Knowing nought of the false wind! Hapless they upon whom thou shined, Thy arts ere they had learned: for in the pictured vow Is shown how I devote, dripping, my garments now, On the wall of his own fane, To the God who commands the main!",
        "author": "Richard W. O’ Brien",
        "date": 1857   },
{
        "text": "What scented stripling, Pyrrha, woos thee now, In pleasant grotto, all with roses fair? For whom those auburn tresses bindest thou With simple care? Full oft shall he thine altered faith bewail, His altered gods: and his unwonted gaze Shall watch the waters darkening to the gale In wild amaze: Who now believing gloats on golden charms; Who hopes thee ever void, and ever kind; Nor knows thy changeful heart, nor the alarms Of changeful wind. For me, let Neptune’s temple-wall declare How, safe-escaped, in votive offering, My dripping garments own, suspended there, Him Ocean-King.",
        "author": "William Ewart Gladstone",
        "date": 1858   },
{
        "text": "What youth, O Pyrrha! blooming fair, With rose-twined wreath and perfumed hair, Woos thee beneath yon grotto’s shade, Urgent in prayer and amorous glance? For whom dost thou thy tresses braid, Simple in thine elegance? Alas! full soon shall he deplore Thy broken faith, thine altered mien: Like one astonished at the roar Of breakers on a leeward shore, Whom gentle airs and skies serene Had tempted on the treacherous deep, So he thy perfidy shall weep Who now enjoys thee fair and kind, But dreams not of the shifting wind. Thrice wretched they, deluded and betrayed, Who trust thy glittering smile and Siren tongue! I have escaped the shipwreck, and have hung In Neptune’s Fane my dripping vest displayed With votive tablet on his altar laid, Thanking the Sea-God for his timely aid.",
        "author": "Henry T. Liddell",
        "date": 1858   },
{
        "text": "What slim youth, whose love-locks flow Wet with unguents, courteth thee, Pyrrha, where the roses blow, And the rocks cool shadows throw On the grotto-floor below? Tell me, tell me, who is he For whom now thou binds thine hair Hair of gold, no witchingly, With that artful careless air! Ah, how oft he shall bewail Broken vow and gods estranged! Unaccustomed to the gale Blackening the erst sunny sea, Marvel that the sea is changed! He who now so trustingly Finds in thee a golden joy, Ever lovely, ever free, From love of all save him for ever Poor silly boy! Hopes that thou– that thou– wilt be; And thinketh never How soon arise Fiercest storms in fairest skies! Wretched they, to whom thou seemest Bright for aye, as now thou gleamest! Thou no more hast power o’er me Votive slab on sacred wall Tells how I most gratefully, To the God who rules the sea, Hung my drippling garments there; For he listened to my call, Ere I sank, he heard my prayer I no longer think thee fair!",
        "author": "Richard Rowe",
        "date": 1858   },
{
        "text": "Oh! Pyrrha, in some grotto hid, Both lying on a couch of roses, Who sports thy wanton charms amid, What amorous youth with thee reposes? What poor fellow, steeped in wine, Mid thy golden curls doth twine Wreaths of flowers, with dainty fingers, While his lip mid nectar lingers? Why so deceive him, heartless one, Inconstant as the fickle sea, Whose charms a thousand beaux have won, Why make him thus confide in thee? Wretched they who know thee not, Theirs a most unhappy lot Wo to him who first hath met Thee, the heartless, and coquette!",
        "author": "J.A. Turner",
        "date": 1858   },
{
        "text": "What stripling, Pyrrha, lavish of perfume, Enraptured woos thee, mid the rosy bloom Of thy delicious grot? For whom that braided knot, That simple, crisp attire? How oft the youth Shall weep thy altered mood and broken troth, Shall dread, when storms arise, The fury of thine eyes. Who hugs the counterfeit as real gold, And hopes, vain hope, the favoring breeze will hold, He ever welcome, thou Serenely kind as now. Fond dupe, whom treacherous calm to shipwreck leads! Long since, on Neptune’s wall, my draggled weeds And votive scroll record The power of ocean’s Lord.",
        "author": "Robert M. Hovenden",
        "date": 1859   },
{
        "text": "Pyrrha, what slender boy, in perfume steeped, Doth in the shade of some delightful grot Caress thee now on couch with roses heaped? For whom dost thou thine amber tresses knot With all thy seeming-artless grace? Ah me, How oft will he thy perfidy bewail, And joys all flown, and shudder at the sea Rough with the chafing of the blust’rous gale, Who now, fond dreamer, revels in thy charms; Who all unweeting how the breezes veer, Hopes still to find a welcome in thine arms, As warm as now, and thee as loving-dear! Ah, woe for those, on whom thy spell is flung! My votive tablet, in the temple set, Proclaims that I to ocean’s god have hung The vestments in my shipwreck smirched and wet.",
        "author": "Theodore Martin",
        "date": 1860   },
{
        "text": "Pyrrha, what tender youth may now With odours sprinkled, urge the vow ’Mid roses in some pleasant cave? For whom is bound your yellow hair With your own neat and simple care? Oft will he view the cruel wave By black winds raised, oft wondering mourn For Faith and Love without return. Who now has won you, trusting boy, Believes you, golden in his joy: Hopes, ignorant of storms, to find You ever his and ever kind. Hapless are they for whom you shine Untried. The mighty Sea-god’s shrine Shews, in my votive tablet there, My dripping clothes suspended were.",
        "author": "Christopher Hughes",
        "date": 1860   },
{
        "text": "What tender youth thee presses in his arms, Perfumed, on roses, revels in your charms, Pyrrha? for whom binds up thy tresses sheen, In cooling grot, so simply neat and plain? How oft that youth your fickleness shall weep, Wonder, his joys what angry whirlwinds sweep, Who blest in fond credulity of love, Hopes thou shalt always amiable prove! Ah! wretched they, for whom thou shinest so fair, Beneath the rose a dangerous serpent there. My votive offering in Neptune’s cell, Shows I’ve escaped your witcheries– farewell.",
        "author": "William Lee",
        "date": 1860   },
{
        "text": "What slip of youth is wooing thee, Bedewed in rose-perfumery, Sacred that hair to whom Simple in golden bloom Syren with artlessness? Ah! he who clasps thee fondly now, Shall often mourn thy broken vow, With tearful eye a-grieving; Gazing on stormy sea, Wailing the Fates’ decree, Fool for believing! Fool not to know the fickle breeze, But ever think to love and please. Ah Pyrrha, thy sweet devilries, Beguile the young untried: I, who fell in Love’s sea, Swam out full speedily. Lo! within the Sea-god’s shrine Hang those dripping clothes of mine Sign that the world may see How, of Pyrrha’s witchery, escaped I the tide.",
        "author": "George Herbert Trevor",
        "date": 1860   },
{
        "text": "What Youth, ’mid roses’ lavish sweets reclining, Now courts Thee, Pyrrha, ’neath some pleasant shade; For whom dost Thou, those locks of gold confining, In simple beauty bind the circling braid? Alas! how oft of fickle Faith complaining, And Gods inconstant, shall he view, aghast, Those waters, late a halcyon calmness feigning, Now black with storms and raging with the blast; Who, dupe to joys that crown thy first possession, Familiar only with Love’s golden hour, Sees peace and pleasure reign in bright succession, Heedless how soon the fav’ring gale may lour! Ah! hapless they Thy syren smiles believing! A votive tablet decks the sacred Fane, And speaks the danger whence the God relieving, I hung my garments reeking from the main.",
        "author": "E.J., Kendal Mercury",
        "date": 1860   },
{
        "text": "Ah, Pyrrha– tell me, whose the happy lot To clip thee on a couch of lavish roses Who, bathed in odorous dews, in his fond arms encloses Thee, in some happy grot? For whom those nets of golden-gloried hair Dost thou entwine in cunning carelessnesses? Alas, poor boy!– who thee, in fond belief, caresses, Deeming thee wholly fair? How oft shall he thy fickleness bemoan When fair to foul shall change– and he, unskilful In pilotage, beholds– with tempests wildly wilful The happy calm o’erthrown! He, who now hopes that thou wilt ever prove All void of care, and full of fond endearing, Knows not that varies more, than Zephyrs everveering, The fickle breath of Love. Ah, hapless he, to whom– like seas untried Thou seemest fair! That my sea-going’s ended My votive tablet shows, to those dark Gods suspended, Who over the waves preside.",
        "author": "Thomas Hood",
        "date": 1861   },
{
        "text": "Now tell me, Pyrrha, tell me sooth, Who is the dainty, perfumed youth Who in that grot, with thee reposes, Upon a bed bestrewed with roses? For whom braids thou thy golden hair, In bright simplicity so fair! Alas! how oft, will he deplore The perfidies thou hast in store! What storms may come, he little knows, When the wild gust of passion blows! Nor dreams of anger, rage, and riot, While he enjoys thee, here, in quiet. He thinks that thou will prove, ever, kind, Nor change like the inconstant wind! Ah! woe to those who, thus, confide In beauties they have never tried! I, as a votive offering, made To the great Sea-God, here have laid My dripping robe to indicate That I’ve escaped so sad a fate.",
        "author": "An Idler",
        "date": 1861   },
{
        "text": "What graceful boy thy form caresses, In the cave so cool, O Pyrrha, With the fresh perfume on thy tresses, White shoulders bathed in yellow hair, On thy couch all blushing with roses rare, Chaste in modest adornment, Pyrrha? Alas! how oft shall his tears deplore, God’s averted and dusky sea, Perfidious flight, desertion sore, Who trustingly thy golden shape embraces, Credulous, all fear away now chases, In passionate clasp now holdeth thee! Thanks to the kind gods, my heart is free: My dripping garment hanging yet On Neptune’s fane by the sounding sea, With votive tablet many a wretch now tells, Whose lovesick soul still soundeth weary knells, Safe I fled the syren’s luring net.",
        "author": "Anonymous, The Harvard Magazine",
        "date": 1861   },
{
        "text": "What slender youth, on rosy couch reclining, Breathing sweet odours, courts thee, Pyrrha, now Beneath some pleasant grot? For whom dost thou, With simple grace thy golden tresses twining, Put forth thy beauty? Ah, how oft shall he Thy broken faith, and Gods estranged, bewail; And see, aghast, beneath the darkling gale, The unwonted ruffle of the angry sea, Who now, confiding, revels in thy charms, Deeming thee purest gold! and hopes that thou Shalt still remain as he beholds thee now, As kind, as open to his longing arms, Nor knows the breeze how fickle! Hapless those On whom thou shines untried! To Ocean’s King How I, escaped, my dripping garments bring, On Neptune’s wall my votive tablet shows.",
        "author": "Edward Smith-Stanley",
        "date": 1862   },
{
        "text": "Pyrrha, in some sequestered grot, Where roses fall around thee, And on thy cheek are kisses hot, What boyish love hath bound thee, His own dark hyacinthine locks With thy fair tresses blended? Ah! quickly come the varying shocks That tell of passion ended, And he will mourn vows light as air, And Pyrrha’s troth departed, Ill-starred, to love a girl so fair, And yet, so faithless-hearted. And I, who watch his ruin, I, Thank Heaven, at Neptune’s door That I have hung My Hat to dry, And tempt the waves no more.",
        "author": "G. Chichester Oxenden",
        "date": 1862   },
{
        "text": "Now, Pyrrha! say what graceful youth Importunes thee amid the roses, The while with liquid odours blessed, He all thy gentle charms discloses? For whom dost thou, so simply neat, Bind up thy wealth of yellow tresses; Or in the coolness of the cave Enrich him with thy soft caresses? Alas! the sailor in that sea Will mourn thy faith so lightly plighted; Will wonder at the rising storms, By blackening winds at once benighted. Who now enjoys thee, golden maid! And hopes to find thee always willing. Oh! luckless they on whom thou smiles With studied art, so sweetly killing! Unhappy they! more happy me, Who now am safe from being stranded I hang my garments on the bank And thank the gods who have me landed!",
        "author": "William Black",
        "date": 1862   },
{
        "text": "What slender youth, besprinkled with perfume, Courts you on roses in some grotto’s shade? Fair Pyrrha, say, for whom Your yellow hair you braid, So trim, so simple! Ah! how oft shall he Lament that faith can fail, that gods can change, Viewing the rough black sea With eyes to tempests strange, Who now is basking in your golden smile, And dreams of you still fancy-free, still kind, Poor fool, nor knows the guile Of the deceitful wind! Woe to the eyes you dazzle without cloud Untried! For me, they show in yonder fane My dripping garments, vowed To Him who curbs the main.",
        "author": "John Conington",
        "date": 1863   },
{
        "text": "For what slight youth on roses laid, In cool grot, dost thy fair hair braid, Pyrrha? Who perfumed folds thy waist, Robed in simple neatness chaste? Amazed, he bitterly shall weep, To find thee angry as the deep. Who now fond credulous in love, Hopes thou wilt always gentle prove. Siren, wrecking all you lure, Unsuspicious, to your bower, My offering, in Neptune’s cell, Shows I have ’scaped your guile– Farewell.",
        "author": "William Lee",
        "date": 1863   },
{
        "text": "What slender stripling courts thee now On roses, ’neath the hanging rocks, For whom, oh Pyrrha! bindest thou, With artless art, thine amber locks? Ah me! how oft shall he deplore Fickle gods and veering truth, Who now is fondly counting o’er Thy charms, all gold,– deluded youth! With what amazement shall he see The waters rough with blackening wind, Who dreams that thou wilt ever be, As now, accessible and kind. Poor souls! they little know their doom, They reck not of the breeze to be, The inexperienced wights for whom Thou shinest like a summer sea. For me,– my votive garments, hung On Father Neptune’s dripping walls, Proclaim that I, when very young, Nigh perished in those fatal squalls.",
        "author": "Anonymous, The St. Andrews University Magazine",
        "date": 1863   },
{
        "text": "What graceful youth with roses crowned, And sweet with perfumes rare, His suit prefers? For whom hast bound Thy braids of golden hair, With simple neatness clad? Ah, me! How often changing skies, And gathering storms he’ll weeping see, With wonder and surprise, Who now enjoys the golden year, Nor fitful breezes knows, But ever free and ever dear Expects thee! Woe to those, On whom thy sweetest smiles are cast; This votive offering given, Plainly declares for me at last, The danger’s over, thank Heaven.",
        "author": "George Howland",
        "date": 1865   },
{
        "text": "What graceful youth caresses thee, Pyrrha, within thy grot’s cool shade; Thy rosy couch spread lavishly, Thy flowing perfumes round him shed? For whom binds thou thy golden hair In elegant simplicity? Alas! how oft will he forswear His faith, and fickle deity! For raging storms shall lash the sea, ’Wildering the unaccustomed youth Who now, all fondly trusting thee, So fair, so bright, believes thy truth Knowing no faithless gale in store, Enjoys thee, hoping thou wilt be As now thou art, for evermore, For him still lovely, and still free. Wretched is he to whom thus seems The untried deep so tempting fair! The mighty Sea God’s shrine proclaims, By votive tablet, hung up there As a remembrance against thee, Of shipwreck suffered so by me.",
        "author": "James Netherby",
        "date": 1865   },
{
        "text": "What stripling, in ambrosial grot, Mid rosy wreaths, that form caresses, For whom fair Pyrrha’s fingers knot In simple charm, those sunny tresses? Alas, how many a tear he’ll shed, When guardian Gods shall faithless flee, And storms unlooked for, overspread The dark ware of his destiny. When she, he deemed the softest-souled, And truest-hearted, leaves him lonely, To learn that all he prized as gold, Was but the glittering surface only. Ill-fated they who see thee fair, With reckless trust believe thee true, Then find the heart that others share, The sport of every wind that blew. To Neptune’s sacred fane did we With votive step the tablet bring, Where all, the pictured garb, may see A saturated offering.",
        "author": "Hugo Nicholas Jones",
        "date": 1865   },
{
        "text": "Pyrrha say, for what stripling, so scented and slender, In thy grot, strewn with roses, all blessing, and blest, Are those ringlets of thine, in the sunniest splendour Of Nature’s embellishment, simply drest: How oft shall he rue The Gods untrue, Love’s gentle horizon, with clouds overcast, Over its waters, alas, That the storm should pass That the girl, in whose faith, like pure gold from the mine, By no falsehood alloyed, he believed to the last, Where affection and constancy seemed to entwine, Should be fickle and false, as the shifting blast! Alas for the peace of the ill-fated lover, Whose heart those untoward enchantments have won, Who thus, when too late, shall be doomed to discover, How false was the charm that had lured him on: No longer to glide Over love’s calm tide, Over the wreck of whose hopes, the dark wave flows, As drifting over Life’s dreary shore A tablet of mine, to the God of the ocean, On the wall of his temple, now votively shows The offering I’ve hung, in my grateful devotion, To hallow his shrine with my streaming clothes.",
        "author": "Hugo Nicholas Jones",
        "date": 1865   },
{
        "text": "What youth crowned with roses, And bathed in perfume, In thy grotto reposes, Oh Pyrrha! for whom, With a charm all so artless, Those bright locks are twined, Soon to mourn thee all heartless, And false as the wind. By the Gods when forsaken, ’Twill burst, like the roar Of dark storms, when they break on Seas, tranquil before; Then, alas! for the lover That trusts to thy smile, Ere his bosom discover How yours could beguile. In thee, he deemed ever, True gold he should find That that heart should be never Untrue or unkind. To the sea’s mighty master, Votive garments of mine, Record my disaster, All dropping with brine",
        "author": "Hugo Nicholas Jones",
        "date": 1865   },
{
        "text": "Say who is that slim little fellow, That clings with such urgent embrace, For whom, Pyrrha, those ringlets of yellow Are twined with so simple a grace, In that grotto, where brightest of roses Around him have lavished their bloom, In whose dreamy seclusion, reposes The fond youth, bedewed with perfume. Alas! for the Gods that could alter, The heart whose affection could fail, How he’ll weep for the faith that could falter, The love that could veer like the gale, When winds are at war with the ocean, As that bosom (oh! strange to behold,) With the one, whose confiding devotion Mistook all that glittered for gold. From the wreck, which the wild wave hath drifted, Thus much is he fated to find, That the love, he thought ne’er could have shifted, Hath left but its sorrows behind. In the temple, a tablet is showing My vows to the God of the sea, And the brine, from my wearables flowing, That speaks of my peril and thee",
        "author": "Hugo Nicholas Jones",
        "date": 1865   },
{
        "text": "In goodly grot, mid roses rare, What dainty lover Pyrrha presses? For whom dost bind thy golden hair With happy artlessnesses? Thy falseness shall he yet deplore, Thee and the changed godlike forms: His inexperience marvelling more At the rough rising storms, Who, credulous, believes thee good, And, heedless of the sudden gale, Holds thee immutable of mood, The fairest of the frail. They are accursed in thy thrall That deem thee true because untried, But I have ’scaped the shipwreck all, Who trusted to the tide. My dripping garments one and all I give the great god of the sea: A tablet on the holy wall I gratefully decree.",
        "author": "Anonymous, The Orchestra",
        "date": 1865   },
{
        "text": "The slim beperfumed boy in store Of roses laid along the floor Of cool inviting grot, Who is he, Pyrrha? for ’tis you He urges to the rendezvous, And asks– Why come you not? Meanwhile for whom give you– for him? Those sunny locks their careless trim, The backward toss and tie? Whose eyes are to be ravished?– his? By that new dress, simplicity’s And neatness’ type, you try? Poor innocent! how oft he’ll weep That gods to kindness cannot keep, Nor you to word you passed: At heaving sea, at whistling loud The winds, and blackening all with cloud, How stand and stare aghast, Who hoped because the prize of gold Was for a moment his to hold, To have you vacant aye, Aye amiable: not aware How like a thing to change is air, And beauty false to play. Miserable they to whom you shine Without experience of the brine! The ruler of the sea Has on his wall suspended yet The votive tablet and the wet Habiliments from me.",
        "author": "Charles Stephens Mathews",
        "date": 1865   },
{
        "text": "What slender stripling in your rosy grot, His locks with liquid perfume shining, Embraces Pyrrha? Say for whom that knot Her golden tresses is confining, Enrichment shunning? but how oft, alack, He’ll mourn her troth and gods invoked forsaken, And stand astounded, by the storm-cloud black And angry billows sudden overtaken. He dotes confiding in your glittering charms, And hopes that, disengaged and still engaging, You’ll always bid him welcome to your arms, Nor thinks the storm will soon be raging. Ill luck for those you dazzle unawares! For I– ’tis on the votive tablet noted In Neptune’s temple, rescued from your snares, My shipwrecked rags, in thanks, devoted.",
        "author": "James Walter Smith",
        "date": 1867   },
{
        "text": "In thy grotto’s cool recesses, Dripping perfumes, lapped in roses, Say, what slender youth reposes, Pyrrha, wooing thy embrace? Braids for whom those tawny tresses Simple in thy grace? Ah! how oft averted heaven Will he weep, and thy dissembling, And, poor novice! view with trembling O’er the erewhile tranquil deep, By the angry tempest driven, Billowy tumult sweep, Now who in thy smile endearing Basks, with foolish fondness hoping To his love thou’lt e’er be open, To his wooing ever kind, Knowing not the fitful veering Of the faithless wind. Hapless they, rash troth who plight thee! On the sacred wall my votive Picture, set with pious motive, Shows I hung in Neptune’s fane My wet garments to the mighty Monarch of the main!",
        "author": "D. A. C., The Round Table",
        "date": 1867   },
{
        "text": "What comely youth ’mid many a row, bestrewed With dripping perfumes, plies his suit to thee, Pyrrha, beneath some pleasing grotto’s shade? For whom dost thou, in artless neatness clad, Uploop thy tawny hair? Ah me! how oft Thy broken vows, and deities estranged, Shall he bewail, and wonder, all-unused, To see the level beauty of thy face Ploughed deep with passion, as the ocean glooms In darkened ridges when the tempest blows, Who now enjoys thee at thy golden time, Too fondly trustful of thy untried worth, And all unconscious of thy wheedling airs, Whole-hearted ever, lovable for aye, Hopes thou wilt be. Ah! pitiable they On whom, unknown, thy beauty glitters keen. For me, vowed tablets with my dangers writ, The sacred wall with dripping garments hung, Proclaim my worship to the mighty God Who saved my youth from shipwreck on thy charms.",
        "author": "Anonymous, The Australasian",
        "date": 1867   },
{
        "text": "Oh, who is the stripling so scented and slim, Who now in your pleasant grot, Pyrrha, reposes On litter of roses, Still cooing and wooing? Those tresses of gold you have braided for him, With charming simplicity! ere very long, For all he is now so confiding a lover, He ’will surely discover Sad treason in season, The smooth waters ruffled by breezes so strong. Fond fool! he believes you as sterling as gold, And trusts he will find you for ever as tender, As prone to surrender; Not ruing what’s brewing, Alas! for the wights who have not known thee of old. The walls of the temple bear witness for me, Who hung up my raiment just after one dipping, All soaking and dripping; My motive was votive; My thanks they were due to the God of the Sea.",
        "author": "T. Herbert Noyes",
        "date": 1868   },
{
        "text": "What graceful youth, in thickest roses’ bower, Wet with his scents woos now free Pyrrha’s hour In pleasant cave’s retreat? For whom so simply neat Braids auburn hair? alas! how oft shall he Mourn altered vows and altered gods, and see With blank amazement strange Rough winds the dark seas change, Who now, all trust, enjoys thy golden smile! And thee still vacant, thee still kind the while Dreams, for he never knew The fickle breeze; fond crew, To whom thou shines all untried! Me my board Votive, on sacred wall to ocean’s lord With vestments hung, portrays Yet dripping from the sprays.",
        "author": "E. H. Brodie",
        "date": 1868   },
{
        "text": "What graceful youth, with odours sweet Bedewed, in some rose-bower’s retreat, Thee, Pyrrha, where its roses meet Caresses? Looking modestly, In fairest rare simplicity, For whom is bound thy golden hair? Alas! how often, in despair, Shall he bewail his destiny, And mourn thy broken faith; and high, When sweeps the tempest o’er the sky, Shall marvel at the raging seas, Who, only used to summer breeze, Enjoys thy seeming priceless love; And, trusting treacherous skies above, Still hopes that thou wilt ever be As kind and gentle, frank and free. Oh wretched!– yet untried thy wiles Whom thy fair sunny face beguiles. A votive tablet shows in Neptune’s fane That, ’scaped the perils of a stormy main, Dank garments, dripping as with briny rain, I’ve hung, in homage to the sea-god’s reign.",
        "author": "Wilmot, Dublin University Magazine",
        "date": 1868   },
{
        "text": "Pyrrha, with the yellow hair Which thou tends with taste and care, Dressing simply what is fair, What youth, perfumed, lapt in flowers, In the pleasantest of bowers Courts thee through the love-led hours? Fondly thinking thou must please Golden ever, he foresees Not the shortly coming breeze. When foul weather follows fair, How will he astonished stare, How lament thy altered air! Wretched must thy lovers be: Long ago I quitted thee And gave thanks for being free.",
        "author": "Edward Yardley, Jr.",
        "date": 1868   },
{
        "text": " What ardent youth, now scented and unguented Stretched amidst roses, strewn on the grateful bed, Clasps thee, O Pyrrha, in his arms? Thou unadorned in sweetest simplicity Binding the golden locks, bursting their silken tie, He must, awakened, curse those charms. As the fond mariner, launched on a placid sea, Thinks it will ever so bright and so gracious be, He ignorant of hostile gales. I, shipwrecked mariner, my votive offering hangs With my wet garments where Neptune’s temple stands High hung on wall, with shipwrecked sails.",
        "author": "John Benson Rose",
        "date": 1869   },
{
        "text": "What graceful boy, while fragrance flows, In rippling breaths, from many a rose, Courts thee, O Pyrrha, ’neath the grateful grot, Thy yellow hair entwined in simple knot? Ah me! Alas! with weary tears, For broken faith he’ll mourn in future years, And sorrowing, wonder when he finds The beaming waters lashed by storm clad winds. Poor fool, he fondly trusts the summer air, And thinks the breezes hushed, the prospect fair Always, nor dreads the treach’rous smiles On heaven’s sweet face. O heedless of thy wiles, Thy bright and glittering snares! O happy me, Free from the dangers of that storm-tossed sea; With wave-drenched garments hung to dry, I place my votive tablets now on high.",
        "author": "I. A. J., The Maritime Monthly",
        "date": 1869   },
{
        "text": "What slender youth, bedewed with liquid odors, Courts thee on roses in some pleasant cave, Pyrrha? for whom binds thou In wreaths thy golden hair, Plain in thy neatness? O, how oft shall he On faith and changed gods complain, and seas Rough with black winds and storms Unwonted shall admire! Who now enjoys thee credulous, all gold, Who always vacant, always amiable Hopes thee, of flattering gales Unmindful. Hapless they T’ whom thou untried seems fair! Me, in my vowed Picture, the sacred wall declares t’ have hung My dank and dropping weeds To the stern god of sea.",
        "author": "Edward Bulwer-Lytton",
        "date": 1870   },
{
        "text": "What slim youth dripping with perfume, In pleasant grot where roses bloom, Woos Pyrrha now to love? For whom Binds thou thy auburn hair In simple loveliness? Ah! me, False gods, faith broken, speedily He’ll mourn, black winds and stormy sea, Who does not look to bear. He now takes all thy coin for gold; He hopes thy whim for aye to hold; Nor dreams of being in the cold. Oh! how I pity all Who know not thy false glitter; I, From shipwreck saved, in memory, A picture and my clothes to dry Have hung on Neptune’s wall.",
        "author": "Thomas Charles Baring",
        "date": 1870   },
{
        "text": "What graceful youth, moist odours breathing, ’Neath some sweet grot with roses fair, Woos thee Pyrrha? For whom art wreathing, With simple taste, thy golden hair? Ah! many a time thy guile deploring, And gods estranged, he shall be seen To view with wonder tempests roaring O’er seas that once so bright have been. Now in thy loveliness confiding, Unheeding thy deceptive breath, He trusts thou wilt be his, abiding Beautiful and true till death. Alas for those unwarned! Thy splendour Is ever but a fickle thing. Lo, on his sacred wall I tender My dripping garb to ocean’s king.",
        "author": "James Griffiths",
        "date": 1870   },
{
        "text": "Pyrrha! what strippling nard-bedewed, Courts thee, in groves with roses strewed? Artless of wordly trick or snare, For whom binds thou thy golden hair? Alas! how oft that youth shall mourn Thy broken faith, and gods forsworn! Who, when black storms upheave our seas, All inexperienced, praises these! Who blindly fondles thee as gold, Trusting no rival to behold; While hoping thou mays faithful prove, Unconscious he, of fickle love! Hapless those lovers everywhere, To whom, untried, thou seemest fair! The sacred wall declares that here, My dripping garments all appear, As votive tablets hung by me, To grace the Ruler of the Sea.",
        "author": "J. O., The Dartmouth",
        "date": 1870   },
{
        "text": " Pyrrha, what slender youth is he Who now reposes, In some delightful cave with thee On beds of roses? For whom, with careless grace arrayed, Dost thou thy golden tresses braid? How oft thy faith will he deplore No longer true, And Gods, alas! the same no more And wondering view, To such a sight unused, the seas Rough with a dark and threatening breeze; Who now most credulously blind Enjoyeth thee Above all price, and hopes to find Thee ever free And ever pleasing, unaware Of winds deceitful. Wretched are Those upon whom thou shines unknown, The sacred wall With votive tablet thereon shown, Proclaims to all That I on Ocean’s mighty God My dripping garments have bestowed.",
        "author": "Mortimer Harris",
        "date": 1871   },
{
        "text": "With liquid odours all bedewed, In some sweet grot, with roses strewed, What slender youth, O Pyrrha, now With thee reclines? For whom dost thou, With careless elegance arrayed, Thy lovely golden tresses braid? Alas how oft thy faith will he And altered Gods weep bitterly; And, ignorant, wonder at the seas Ruflled by an unfavoring breeze, Who now, with credulous fondness filled, Enjoys thee priceless; and unskilled In falsehood’s breath hopes still to find Thee ever free, and ever kind! Ah! wretched they on whom untried Thou shines. The sacred wall supplied With votive tablet shows that I Have hung my garments not yet dry To Ocean’s potent Deity.",
        "author": "Mortimer Harris",
        "date": 1871   },
{
        "text": "O Pyrrha! decked with roses, What slender, perfumed boy His eager suit discloses, And all his trust reposes, And all his new-born joy, On you, so sweetly coy! You bind your sunny tresses With artless art for him; Alas! he little guesses The fickle faith that presses, Instant, and rude, and grim, Your seeming gold to dim. The roughening waters round him, He neither sees nor hears; How will the winds astound him! The strong black winds that found him, With murmurs in his ears, That boded not of tears. He thinks her his for ever, With store of love for both; Alas! for those who never Distrust, until they quiver With broken heart and troth, And many a shattered oath. He thinks that nought can change her; My dripping garments see, Which I, a shipwrecked stranger, But snatched from death and danger, Vow to the god of sea, Whose strong hand rescued me.",
        "author": "M. C.",
        "date": 1871   },
{
        "text": "Ah! Pyrrha! say what slender boy Is this thy fleet young form pursuing? Helter skelter in his joy O’er myriad roses to thy wooing; All drenched in fragrance, limbs, and head, With dews from shaken rose cups shed, And whom thou sooths to calmer bliss, In yon cool cavern with a kiss; For whom binds thou thy yellow hair So artless! in these homely braids? For whom? alas! all wrung with care; He oft must mourn a faith that fades, And gods estranged, yet still admire The fitful fury of the sea; Storm-lashed, and pale with foaming ire, And praise the wind so wildly free; ’Tis he, the same, whose boyish faith Grows great with thee his golden dream, And, unsuspecting, fondly saith; She is– all other maids but seem! Unhappy they that worship thee! To whom untried thou’rt falsely fair, Ah! bid them view yon wall and see My votive tablet hanging there, Which tells in grateful words of mine, How, saved from shipwreck on her shore, I hung those garments wet with brine, Vowed to the Sea-God ever more.",
        "author": "G. G. M’C., The Australasian",
        "date": 1871   },
{
        "text": "Ah! tell me, sweet Pyrrha, what beautiful boy This evening shall feast on your charms, In some bower ’mid the roses, the scene of your joy, In the raptured embrace of your arms. For whom are you binding your gold-gleaming hair, And so simply adorning your dress? Shall this lord of your fond love be doomed to despair When a shipwreck his barque shall oppress? Shall he who believes you the best of the best, And dreams not of change and deceit, Shall he who reposes his head on your breast Find all your endearments a cheat? How I pity the wretch who descries not the snare That beams from your eye and your brow! Escaped from the wreck of my peace, now, I swear, A fit offering to Heaven I’ll vow.",
        "author": "D. C. L., The St James’s Magazine",
        "date": 1871   },
{
        "text": "Amid the roses in pleasant shade, What youth is he Who, Pyrrha, steeped in odours sweet Caresses thee? For him you bind your golden hair So neatly plain. Alas! how he shall mourn thee lost, And mourn in vain; And mourn the Gods for ever changed And thee forsworn, And see anon the roughening seas With tempests torn, Who now in bliss, believing thee In bliss, remains Thinks thee all gold, thy pure heart free From other chains. Oh, Neptune! in thy briny halls Beneath the sea My votive tablet I have hung, Great God, to thee.",
        "author": "George Augustine Stack",
        "date": 1872   },
{
        "text": "What youth with rose-crowned brow, O Pyrrha, woos thee now, Sprinkled with odours sweet In some cool retreat? For whom thy yellow hair Dost thou bind with care And simple grace? alas, He through storms must pass. How oft in blank amaze, He to his gods must raise Plaints of thy false vows And their clouded brows, Who now thy heart doth hold Thinking it pure gold, Hoping thee to find Ever true and kind! Ah! little does he know How the winds will blow; Most unhappy he Who trusts that sparkling sea. For me, my pictured woe The sacred walls may show; My dripping robes I bring, The sea-god’s offering.",
        "author": "Louisa Bigg",
        "date": 1872   },
{
        "text": "What stripling, Pyrrha, lavish of perfume, Enamoured wooes thee, mid the rosy bloom Of some cool grot reclined? For whom dost neatly bind Thy tresses unadorned? Oft shall the swain To careless Gods of broken vows complain, And view in strange surprise Rough seas and blackening skies. He hugs the counterfeit as real gold, And hopes, vain hope, the favouring breeze will hold, He ever welcome, thou Serenely kind as now. Fond dupe, whom specious calm to shipwreck leads! For me, on Neptune’s wall, my draggled weeds A warning record keep Of perils in the deep.",
        "author": "Robert M. Hovenden",
        "date": 1874   },
{
        "text": "Say, Pyrrha, what fine youth it is, With roses crowned, loves you to kiss, His Pyrrha, in some grot? For whom is it you knot, So sweetly neat, your yellow hair? Ah, well, to him the skies seem fair! But how long will they seem? It is a pleasant dream He dreams, no doubt: he deems you true: No doubt he finds a world in you Of love and that!– How strange, If fickle winds should change, It all will seem to him!– For me, I think I must most lucky be, So barely I escaped A Circe goddess-shaped!",
        "author": "Thomas Ashe",
        "date": 1874   },
{
        "text": "What slender youth on bed of roses, Pyrrha, by thy side reposes, With odours perfumed sweet In shady grot reclined? And when her waving auburn tresses With neat simplicity she dresses, Oh, whom is it to greet? For whom art thou so kind? Alas, how oft will that fond boy Who now so blindly can enjoy Thy venal beauties, weep Thy broken vows of love, When all thy perjury he finds; And wondering at the roughening winds, That brush the darkling deep, Will woman’s folly prove: Hapless,– he knoweth not thy wiles, But hopes to bask in all thy smiles, And have thee his alone; Still, those are more unblest, Who all in vain thy charms approve; For me half-drowned in Pyrrha’s love Before Old Neptune’s throne I hang my votive vest.",
        "author": "Martin F. Tupper",
        "date": 1874   },
{
        "text": "What slender youth on beds of roses, Drenched with many a perfume sweet, Embraces Pyrrha in a cave? But she her yellow hair disposes With all the neatness that is meet. How oft the gods who will not save, And change of faith, will he lament, And wonder at the blackening wave. He stormy, and she insolent, Who loves you thinks you best of all; Hopes you will true and kind remain, Unconscious of the faithless breeze; Unhappy in whose way you fall Untried, behold, in Neptune’s fane My garments, dripping from the seas, Suspended on the sacred wall.",
        "author": "Charles H. Hoole",
        "date": 1875   },
{
        "text": "What dainty youth, with dewy odours spent, ’Neath many a rose within some pleasant bower, Lady, solicits thee:– For whom Braided is thy bright hair? Nice in its negligence!– how oft, alas! Shall he of mutable faith and fate complain; And wonder at the darkness strange Of the storm-fretted deep; He, who now revels in thy wealth of love, Deeming thee all his own, and ever kind; Unconscious of the fitful gales Ill-fortuned they, on whom Untried they smile!– For me, the chapel wall Suspended on a votive tablet shows My sea-drenched garments, dedicate To Him who rules the main.",
        "author": "Edmund Lenthal Swifte",
        "date": 1875   },
{
        "text": "What perfumed boy beside you now reposes In some cool shade, with eager, mad caresses; While you, to please him, ’mid the dropping roses, Let fall your golden tresses? Artfully artless! how the child will wonder, When this fair day of love, so bright, so warm, With black clouds overcast, and bursting thunder, Shall change to sudden storm! Facile and tender when her whim it pleases, He thinks, fond fool, this golden hour will last; But sooner hope to fix the faithless breezes Than hold her to her past! But I, experienced in each subtle motive Which brings such shifting gales o’er love’s wild sky, Hang in the Temple, as an offering votive, My sea-drenched panoply.",
        "author": "J. F. C. and L. C., Exotics",
        "date": 1875   },
{
        "text": "Say, Pyrrha, lovely maid, What tender youth reclining at thy feet Fragrant with liquid odours sweet, Basks in thine eyes’ soft lustre, In that cool grotto’s shade Where roses cling and cluster? For whose enraptured eye Dost thou those sweet neglectful fingers ply, That into careless plait have braided Each golden yellow tress Thine own unstudied loveliness, Which no false art has aided? Alas! how oft he’ll weep Thy broken vows, and fickle gods bemoan, Propitious once, now hostile grown! How oft (yet new to unrepaid devotion), He’ll wonder at the storms that sweep O’er Love’s once-tranquil ocean! Who, finding thee all fair, All smiles to-day, still hopes– ah, too confiding! Thy love, thy charms will prove abiding, Nor dreams that ere the morrow Will veer the changeful air, And turn his joy to sorrow! Alas! unhappy they For whom Love’s surface smiles, and smiles untried! That I’ve escaped the fickle tide The temple-wall with votive slab declareth; Where, dripping from the fatal spray, Hang dedicate, a lover’s trappings gay To that dread Power which Ocean’s sceptre beareth.",
        "author": "R. D. F. S., The New Monthly Magazine",
        "date": 1875   },
{
        "text": "What slender youth, all bathed in streaming perfumes Presses his suit to thee on piles of roses Pyrrha, beneath some lovely grotto? For whom bindst thou thy golden tresses, Artless in elegance? Ah me, how often Shall he thy falsehood weep, and gods estranged, And gaze with inexperienced wonder On waters rough with storm winds gloomy, Who now, poor dupe, enjoys thee golden-gay, Who hopes that thou wilt aye be fancy-free, Aye charming, knowing not the breeze How faithless! Hapless they, to whom Thou art a radiancy– untried! For me, The consecrated wall by votive-picture Proclaims I’ve hung my drenched raiment Unto the god that rules the sea.",
        "author": "Arthur Way",
        "date": 1876   },
{
        "text": "Who is the slender youth bedewed With perfumes, decked with roses, Who last fair Pyrrha’s charms has wooed, And in some grot reposes? For whom dost bind thy yellow hair So simply and so neatly? How oft at fickle faith he’ll swear, And curse his gods completely! How oft he’ll see– unwonted sight His ocean all o’ercast, Who for a while basks in thy light, And thinks that light will last. He deems thou ever wilt be dear, Thy favours aye the same, Forgetting how the wind may veer Poor moth, unused to flame! For me, I’ve shipwreck escaped; the wall And votive brass declare, I’ve hung my dripping garments all In Neptune’s honour there.",
        "author": "W. E. H. Forsyth",
        "date": 1876   },
{
        "text": "What youth now seeks thy roseate bed, Beneath some time-worn grotto spread? For whom, fair Pyrrha, braid you now Your golden hair across your ivory brow? Alas! how often shall he weep Thy broken faith, thy passion dead? As he has mourned when morning’s tranquil deep Rose to the evening’s storm, and all its beauty fled. For me: my shipwreck’s vow is paid, And, on the votive tablet laid, My storm-drenched garments rest above, A grateful offering to the sea-god’s love.",
        "author": "S. W. Langston Parker",
        "date": 1876   },
{
        "text": " Where myriad roses bloom, Bedewed with rich perfume, What slender boy now woos thee with his pray’r In some fair bower’s shade? Say, Pyrrha, wanton maid, For whom dost thou array thy golden hair With neat simplicity? How many a time, ay me! Will he bewail the gods who look away, Thy fickleness as well, And wonder at the swell (Poor fool!) of the angry waters lashed to spray, Who now with trusting breast Enjoys thee at thy best; Who hopes (unknowing how the wind may turn) That thou wilt ever be All amiable and free! O hapless those who inexperienced burn With longing after thee! But I have sailed the sea, And safely come through tempest home to port! The temple walls declare, With votive picture there, My dripping weeds to Neptune I have brought!",
        "author": "William Johnston Hutchinson",
        "date": 1876   },
{
        "text": "Who, on plenteous roses– stripling slim, Drenched with liquid odour,– sues thee hard In pleasant grot? for whom Tiest thou, Pyrrha, thy auburn hair Simple in grace? How oft, alas! will he Faith and changed gods lament, and soon In strange surprize behold Black winds sweep on a ruffled sea! Now he joys to eye thee golden-bright, Hopes thee alway open, alway kind; Fond fool! of shifting breeze Thoughtless. Woe for the hearts, to which New thou glitt’rest. Me the sacred wall Shows on votive board, when high I hung My dripping weeds;– a gift Gladly paid to the Seagod’s might.",
        "author": "F. J. Newman",
        "date": 1876   },
{
        "text": "What slender youth, with liquid perfumes sweet, In rosy nook, now worships at your feet? For whom is bound that wealth of golden hair Which waves with simple folds in beauty rare? How oft will he who tastes your favors now Bemoan the fates– bewail your broken vow Amazed will be, when darkening clouds arise Who now rejoices in your love-lit eyes Who dreams, before the treacherous gale, that you Will be to him forever fond and true! Unhappy he who, knowing not your wiles, Basks in the glory of your witching smiles. Though calm the sea beneath and clear the sky, The storm will come and billows roll on high. The offerings made at Neptune’s sacred shrine Attest that I have escaped the raging brine!",
        "author": "Thomas Chalmers McCorvey",
        "date": 1877   },
{
        "text": "What slender boy, bedewed with moist perfume, Is wooing thee midst many a rose’s bloom, Pyrrha, ’neath the pleasant grot? For whom thy yellow hair dost knot In simple sweetness? Ah! how oft shall he Thy troth bewail and changed gods, and the sea, Roughened by the gloomy wind, Behold with rapt, unwonted mind, Who trustful now enjoys thy golden hue, Who hopes thee ever free and ever true, Heedless of the treacherous air: O wretched they to whom so fair Thou seems, untried; but I, with tablet vowed, The sacred wall declares, my dripping shroud Have suspended in his fane Unto the god who rules the main.",
        "author": "Caskie Harrison",
        "date": 1877   },
{
        "text": "Who is the graceful polished wight Who now the perfumed kiss may pour On Pyrrha’s lip through half the night Within her pleasant rose-decked bower? For whom dost thou thine auburn tresses So simply, yet with art, arrange? Dreams he how soon those hot caresses, Like tropic gales, will veer and change? Little, fond fool! his heart forebodes The prompt wreck of that plighted troth Troth sealed and sworn by all the gods With perjured tongue and brittle oath! Too well he trusts, poor credulous youth! Thy golden-sometime gold-bought-smile; Reliant on thy steadfast truth So innocent of wrong or guile. The gleam that in thy false eye dances, Now worshipped without doubt or fear The sun-light of those amorous glances, In what dark clouds it will disappear! Alas for those who’ll yet discover The falsehood of thy practised charms! I’m thankful-for my pains are over To scape with no severer harms.",
        "author": "Anonymous, Kent and Sussex Courier",
        "date": 1877   },
{
        "text": "Pyrrha, what slender youth, bedewed With liquid fragrance, t’ward thee presses, In pleasant grot, with roses strewed? For whom, at thy neat toilet wooed, Dost thou bind up thy golden tresses? Alas! how oftentimes will he, Who, full of trust, now fondles thee, And, dreaming not of treacherous wind, Pictures thee ever fair and kind, How often will he mourn at last Thine and the gods’ inconstancy! How, unaccustomed, stare aghast Upon a dark, storm-roughened sea! Woe will abide With them on whom thou shines untried! But that to ocean’s sovereign I have hung up my dripping dress, My tablet in his holy fane, My votive tablet witnesses.",
        "author": "William Thomas Thornton",
        "date": 1878   },
{
        "text": "Who, Pyrrha, now thy favour shares? What slender youth for thee prepares The cool and pleasant grot? What scents and roses strewed around, For whom hast thou thy tresses bound Into its golden knot? Most pleasing in thy simplest dress, Alas! how often will distress, From changing humours flow! With wonder at the sudden rise Of tempests from o’erclouded skies, Whilst gales, unlooked for, blow. He who now seeks thy love to gain Thinks thou wilt ever true remain, And docile to his sway: Unconscious of his fate, poor fool! He little knows thy treacherous school Or thy deceitful way. Unproven thou! unhappy they! Who live on smiles and die thy prey, Whilst I, from damage free, Will votive hang on Neptune’s wall My garments tattered by the squall, And dripping from the sea.",
        "author": "William Knollys",
        "date": 1878   },
{
        "text": "What slip of youth is wooing thee, Bedewed in rose perfumery, In ball-room’s cool recess? Sacred that hair to whom Wanton in golden bloom, Siren of artlessness? Ah! he who woos thee fondly now Shall often mourn thy broken vow, With tearful eye a-grieving; Gazing on stormy sea, Wailing the Fates’ decree Fool for believing! Fool, not to know the fickle breeze, But ever think to love and please; Ah! Helen, thy soft witcheries Beguile the young untried: I who fell in Love’s sea Swam out full speedily; Now as dry as any bone, With a wife of fourteen stone, And a fortune of her own, Bid I my neighbours see How of Helen’s witchery ’Scaped I the tide.",
        "author": "George Herbert Trevor",
        "date": 1878   },
{
        "text": "What slender stripling, bathed in perfumes sweet ’Midst roses thick, thee Pyrrha presses, Within some pleasant cave, for lovers’ meet? For whom bind’st thou thy yellow tresses, So simply neat? Alas! how oft will he Faith and the fickle Gods lament, And tempest-tossed by the black winds the sea, Unused, behold with wonderment, Who now possesses thee, all golden thought, Who ever loving, ever true, Hopes thee to find, of breeze fallacious naught Conceiving! O! what woe’s their due On whom thou shines untried! On sacred wall, The votive picture shows by me My garments have been hung up, dripping all, Unto the God who rules the sea.",
        "author": "James John Lonsdale",
        "date": 1879   },
{
        "text": "What graceful youth perfumed with liquids sweet, Midst showers of roses, in some cool retreat, Woos thee, O Pyrrha? And for whom Dost bind thy hair of golden bloom So plainly neat? Alas! how oft shall he Deplore thy faith and Gods so changed! and be, All inexperienced, amazed At seas, rough with black storms, upraised, Who, credulous, enjoys thy precious charms: Who, trusting in thy love, hath no alarms, All ignorant of the fickle wind, That thou could’st ever prove unkind! Curst those to whom thou, all untried, shines fair! But votive tablets on the sacred walls declare That I my dripping robes decree To Neptune, God who rules the sea.",
        "author": "X. O. C., Weak Moments",
        "date": 1879   },
{
        "text": "What youth, slenderly built, dewy with liquid scents, Courts thee, Pyrrha, amid roses so numerous, In some favorite grotto? For whom braidest thy flaxen hair Thou thro’ neatness so plain? Ah! but how often faith And gods changeable he’ll grieve o’er, and oceans vast Roughed by blackening cyclones He too greatly will wonder at, Who now joys him in thee, golden believing thee, Who expects thee to be ever free, ever thus Love-like, not scenting guile on Love’s sweet breath; How forlorn, for whom Thou, set, seemest to shine! By votive-tablet borne, This wall, sacred, declares me to have hung thereon Vestments dank to the potent God– feared god of the ocean!",
        "author": "John Cutler",
        "date": 1881   },
{
        "text": "What slender youth with liquid scents bedewed Now courts thee, Pyrrha, in sone grot’s cool shade, Upon a couch with rosy garlands strewed? For whom dost thou thy golden tresses braid, Arrayed in simple elegance? How oft, Alas! shall he thy fickleness bewail, Who now, fond fool, exchanges kisses soft; And at rough seas, stirred up by passion’s gale, Shall marvel greatly, who hopes thee to find Ever in sweet abandonment as now; Himself unmindful of the fickle wind Which, springing up, may ruffle Ocean’s brow. Unhappy they, who, by thy smile allured, Have suffered shipwreck. I no more essay The treacherous deep; but, safety now secured, To the Sea-god my willing vows I pay.",
        "author": "John Augustus Miles",
        "date": 1882   },
{
        "text": "What dainty lover steeped in scented air Among the roses scattered everywhere, Shall woo you, Pyrrha, in the grateful cave? Tell me for whom you bind your yellow hair Simple in neatness? Ah! how often he Shall learn to weep for your fidelity And the changed gods, and wondering behold How the black breezes rouse the writhing sea! Alas, poor fool! whose fond and faithful mind Believes you all his own, and always kind Yea, deems you purer than the purest gold, Who are as false and fleeting as the wind. Most wretched he to whom untried you shine; My votive tablet on the wall divine Proclaims for me that long ago I gave My dripping garments to the Sea-God’s shrine.",
        "author": "Justin H. McCarthy",
        "date": 1882   },
{
        "text": "What slender youth perfume bedewed, Courts thee ’mid roses many hued? O Pyrrha, thou art divinely fair. For whom so wreathe thy golden hair, In beauty plain? Oh! how should we Mourn yon faithless gods to see, And wild waters rudely tempest-tossed, Admiring thee, indeed, till lost! Ah, heedless they, all credulous, Who see thee turn and sorrow thus, Unmindful of the rising gale, Hapless and direful in its wail; Untried by time you seem so frail, My tablet’s vow shall yet declare, My dank and dripping weeds to be Given to the sable goddess of the sea.",
        "author": "Jacob Guy Collins",
        "date": 1883   },
{
        "text": "What delicate youth, fickle Pyrrha, enfolds thee, Caresses thy charms in the vine-covered grot, Where many a rose in its fragrance upholds thee, And sprinkles the dew as a gift o’er the spot? Thy tresses of gold, ah! for whom dost thou braid them, So neatly arrayed in thine ornaments few? This credulous boy who believes in thee, maiden, As fancy-free, always thus lovely to view; Nor dreams that the vows thou hast breathed may be broken; That promise of gold may be dross in the hand; That sometimes the sigh of the zephyr, false token, May tell of the gale and of wreck on the strand; How often he’ll weep o’er the hope that is faded; The pledge on thy lip as the dew that is fled; And blame the false gods when the storm heavy-laded Hangs over the deep, as a pall o’er the dead! Ah, pity the heart by thy charms lately smitten! In yonder proud fane is a tablet for me; My garments all dank– as a tale that is written A vow to the god who rules over the sea.",
        "author": "Henry Hubbard Pierce",
        "date": 1884   },
{
        "text": "What perfumed youth on flowers reclined Courts thee within some loved retreat? For whom dost now thy tresses bind Pyrrha! so dainty and so sweet? How oft o’er broken vows he’ll weep, And, cruel fate! amazed, behold Seas dark with winds around him sweep, Who erst believed thee true as gold; Who deems you still so fond and free, Unconscious of the fitful gale, Like others, wretched soon shall be, And mourn you fickle, false, and frail, My storm-drenched garments on the wall, And votive tablet fixed above, The mighty sea-god’s power recall, Who saved me, well nigh wrecked through love.",
        "author": "Herbert Grant",
        "date": 1885   },
{
        "text": "With scents bedewed, what silly boy Pyrrha, now seeks your love t’ enjoy, ’Neath pleasant grot with roses twined, For whom your yellow hair you bind So simply neat? He soon will curse Your broken faith and Gods averse; Artless, he’ll soon astounded be At darksome winds and raging sea; Who, trustful now, enjoys your charms, And hopes that only in his arms You free and loving he shall find, Unconscious of the faithless wind. Hapless are they for whom you shine So bright, untried! The sacred shrine By tablet vowed, suspended shows The sea-god’s gift,– my dripping clothes!",
        "author": "Charles William Duncan",
        "date": 1886   },
{
        "text": "What lissome lad, perfumed with dripping otto, Woos thee, on roses, in thy pleasant grotto, Pyrrha? For whose caresses Dost bind thy yellow tresses, Simple in neatness? Alas, how often Thy fickle love and gods he’ll try to soften In vain, and watch with wonder Thy wind-tost waves and thunder That greenhorn– who, thy golden youth enjoying, Hopes still to find thee fancy free, nor cloying Of amorous sweets, unheeding Thy tricksy gusts. But needing Pity are they whom thou untried hast dazzled. The sacred wall’s vowed picture shows my frazzled, Soaked suit, to thee suspended, Strong Sea God– folly ended.",
        "author": "William Preston Johnson",
        "date": 1886   },
{
        "text": "What gentle youth in cool verandah seats Where sweet exotics weave a grateful shade, Sits sighing at thy feet? For whom are charms so prettily displayed In coy simplicity? Daisy, how soon shall he pause mournfully To gaze on shattered idols, and deplore The tempest’s ravages, unfelt before; Who now contented in the golden blaze Of summer gladness, ever trusts to find Thy fickle heart ingenuous and kind, Unhappy they who love but know thee not! My heart has one bright spot Filled with sweet memories– not small their cost All that survived when all beside was lost.",
        "author": "Anonymous, Homeward Mail from India",
        "date": 1886   },
{
        "text": "What dandy youth, O Pyrrha fair, Bedewed with liquid odours rare, Caresses you in yon sweet grove Enchanted with your smiles and love! Who in your lap supinely doses In midst of jesamines and roses; For whom do you with so much care Thus fillet up your golden hair, Sincerely delicate and pleasing, Nor shy, nor coquettishly teasing! Alas! how frequently shall he Deplore your vile inconstancy, Your perfidy and treacherous dealing, With want of all religions feeling! Through inexperience shall he Astonished at your conduct be To see the sea with storms arise And darkening clouds overspread the sides. The simple youth who in thy smile Basks, and believes thee void of guile, Shall yet meet with a great surprise When he sees through the slim disguise. He who expects you still to be Aye disengaged and always free, In love and beauty still increasing To please and charm him never ceasing, In danger is. He spreads his sail Unconscious of the faithless pale. All such poor silly wretches are For whom you shine so bright afar But as for me, you surely know I ’escaped such shipwreck long ago. To show that I’ve escaped the squall, I’ve hung on Neptune’s temple wall This votive tablet, there to tell That storms and dangers me befell, And that, with gratitude elate, I, therefore, here do consecrate The dripping garments worn by me Unto the god who rules the sea.",
        "author": "Anonymous, Glasgow Weekly Herald",
        "date": 1886   },
{
        "text": "What dainty boy, in rosy wreath And dewed with perfume, dreams an hour, Fair Pyrrha, at thy side beneath The shade of some luxurious bower? For whom that yellow hair dost bind So subtly simple? Oft his eyes Shall weep false faith and Gods unkind, And watch with innocent surprise Black winds o’er roughening ocean swirl, Who clasps, fond fool, his golden girl; Who, witless of the treacherous squall, As open ever and as kind Believes thee. Miserable all On whom thy loveliness hath shined, Yet unessayed! And men may see The marble of the holy shrine On votive tablet picture me Plucked from the perils of the brine, And hanging dripping robes to be Thanks-offering to its deity.",
        "author": "T. Rutherford Clark",
        "date": 1887   },
{
        "text": "Bedewed with odorous balms, what pretty boy, On heaps of roses in some pleasant grot, Pyrrha, with thee doth hotly toy? For whom dost backward knot Thy yellow hair, bewitching simple? Oh. How will he mourn changed gods and broken troth, And stare amazed, when bleak winds blow, And roughened seas are wroth, Who now, fond fool, enjoys thee, deems thee gold, Who, never having known a treacherous breeze, Hopes thee still his, all his to hold, Still loving! Woe for these, On whom thy wiles are newly flung! A votive tablet in his temple shows, I’ve to the sea’s great god uphung My brine-bedabbled clothes.",
        "author": "T. M., The Bookmart",
        "date": 1887   },
{
        "text": "What graceful boy, dripping with rich perfume Woos thee among roses in some grotto’s shade? Pyrrha! for whom Dost thou thy yellow tresses braid In simple neatness artlessly arrayed? Alas, how oft shall he who credulous dreams That all is Truth that truthful seems, Basks in thy sun, nor doubts that he alone Shall ever call thy golden grace his own, Heedless of treacherous gales, and love not tried, How oft bewail thy broken faith, and chide The changeful Gods, and stare with wondering eye On rough seas blackening ’neath a cloud-swept sky! Most miserable they Whom, falsely fair, thou glitterest to betray! I, too, have hung on Neptune’s hallowed shrine My picture vowed, and garments dark with brine To that all-powerful God whom winds and waves obey.",
        "author": "Stephen De Vere",
        "date": 1888   },
{
        "text": "Pyrrha! What graceful youth, declare, Suffused with liquid perfumes rare, Now you in some cool grot caresses, For whom you plait your amber tresses? Ah! Oft shall he with dread surprise, See ocean rough as tempests rise, And for changed Gods and faith shall grieve, Who fondly doth in thee believe, And hopes thou wilt for ever be To him unoccupied and free, In happy ignorance of mind That ne’er forebodes the change of wind. But a sad fate doth them abide To whom, on trust, you shine untried. In Neptune’s fane a tablet shews I there hung up my dripping clothes, A votive warning thence to be Of shipwreck on that treacherous sea.",
        "author": "W. E. Surtees",
        "date": 1888   },
{
        "text": "What swain superb in youthful graces With scented dews embalmed and shining Courts thee, fair maid, with fond embraces On roses in some grot reclining. For whom so coy in plain adorning Dost bind those tantalising tresses Bright as the golden hues of morning? Alas! fond youth, he little guesses. O’er altered fates and pledges weeping, How oft he’ll view with artless wonder The black squall o’er the calmness sweeping The sunny heaven grow dark with thunder. So bright to him the outlook seems Who recks not of the fickle zephyr, And sees nought but thy golden beams, And deems the true and fast forever. Ah! fools the shining harbour seeking, The hidden shoals they’ll soon discover! At Neptune’s shrine those garments reeking, Proclaim my toils and perils over.",
        "author": "Edward Jenkins",
        "date": 1888   },
{
        "text": "What graceful youth on many a rosy bed, With liquid perfumes breathing round his head, For whom thou hast thy locks in simple neatness laid, Embraces thee, O Pyrrha, neath the grateful shade? How oft shall he thy fickle heart deplore, And changed gods! He, at old ocean’s roar, When white-capped waves by blackening storms are tossed, Untaught, shall stand amazed and gaze in rapture lost. He credulous, thy qualities all gold Enjoys, and since of faithless winds untold, Still hopes that in the future thou may’st always be As now, in love so loving and in fancy free. How wretched he whom passion tempts to sip The nectar sparkling on thy ruby lip, Or seek thy other charms: for soon thou’lt heartless prove And show thyself most scornful of his love. Behold in Neptune’s Fane, the sacred wall By votive tablet indicates to all, How I my garments, wet with ocean’s wildering spray, Have hung a sacrifice of love’s lone cast-a-way.",
        "author": "T. M. Lee",
        "date": 1888   },
{
        "text": "What favoured Youth dost thou fair Pyrrha own? Who courts thy smile midst odorous roses fair, In Grotto privileged to thee alone; For whom in wreaths dost bind thy golden hair? Gracefully simple– ah, on changed faith, How long shall he to Sea and Gods complain! With shuddering awe look back on that rude storm Which wrecked his hopes on a too treacherous main. Who trusts thee now? holds thee than gold more pure, Thee! fickle, changeful, weak in loving vain; Heedless of flattering gales and winds which lure, To dash his tossed bark on the strand again. Unhappy they, to whom thou seemest fair, As yet unknown– so art thou not to me; My dank and dripping weeds hung up declare, That peril ’scaped from Neptune and from thee.",
        "author": "E. H. Stanley",
        "date": 1889   },
{
        "text": "What radiant youth, with perfumes rare Besprinkled, now in pleasant bowers Pays court to Pyrrha, crowned with flowers? For whom unbindest thou thy hair In neatness charming? Ah, the tears For broken faith and gods unkind! When billows raised by stormy wind O’erwhelm him, strange shall be his fears Who, credulous, enjoys to-day Thy peerless beauty– thinks thee his, And free for him alone, nor is Of storm impending conscious! They Are lost who trust thee! Neptune’s shrine, With votive tablet decked, displays, Suspended to the sea-god’s praise, My garments dripping from the brine.",
        "author": "E. Hamilton Irving",
        "date": 1889   },
{
        "text": "What graceful youth with waxed moustache, And on his head pomade, Takes thee, oh Pyrrha, out at night To walk the promenade? Who is the gallant youth for whom Thou deck’st thyself so gay? Who is the mighty swell whom thou So fascinates to-day? Some day those high and lofty hopes, Which fill his youthful mind (That thou wilt once be his), alas! Will fly before the wind. And he’ll lament that e’er he saw Thy outward witching form; And never dreamt that from within Would spring a deadly storm; But ’tis the case, for I myself Once loved a lassie fair, A prettier girl I never saw, With downy dark-brown hair, And hazel eyes, and such a chin! But I have had my share!",
        "author": "M. F. H., Liverpool Echo",
        "date": 1889   },
{
        "text": "What slender youth, with perfumed locks, In some sweet nook beneath the rocks, Pyrrha, where clustering roses grow, Bends to thy fatal beauty now? For whom is now that golden hair Wreathed in a band so simply fair? How often will he weep to find Thy pledges frail, Love’s power unkind? And start to see the tempest sweep With angry blast the darkening deep; Though sunned by thy entrancing smile He fears no change, suspects no guile. A sailor on bright summer seas, He wots not of the fickle breeze. For me– yon votive tablet scan; It tells that I, a shipwrecked man, Hung my dank weeds in Neptune’s fane And ne’er will tempt those seas again.",
        "author": "Goldwin Smith",
        "date": 1890   },
{
        "text": "What slender boy, with perfumes sweet, In thine own grotto’s blest retreat, Beneath the rose’s shade, Woos thee, O Pyrrha? for what swain Dost thou thy yellow locks restrain, In simple grace arrayed? Alas! how often shall he weep The plighted faith thou wilt not keep, The Gods themselves grown strange, And gaze with wonder on the deep, By rough winds wakened from its sleep, And marvel at the change! He who enjoys thy golden prime, And trusts, poor fool! in future time To find thee kind and free, Knows not how fickle is the wind, Nor yet how sad a fate they find, Who blindly worship thee! The temple’s sacred wall declares How I, escaped from all thy snares, In votive offering gave, As sailor saved from stormy sea, My dripping raiment up to thee, Great Monarch of the wave!",
        "author": "J. Leigh S. Hatton",
        "date": 1890   },
{
        "text": "What perfumed, posie-dizened sirrah, With smiles for diet, Clasps you, O fair but faithless Pyrrha, On the quiet? For whom do you bind up your tresses, As spun-gold yellow, Meshes that go with your caresses, To snare a fellow? How will he rail at fate capricious, And curse you duly, Yet now he deems your wiles delicious, You perfect, truly! Pyrrha, your love’s a treacherous ocean; He ’ll soon fall in there! Then shall I gloat on his commotion, For I have been there.",
        "author": "Eugene Field",
        "date": 1891   },
{
        "text": "What dainty boy with sweet perfumes bedewed Has lavished kisses, Pyrrha, in the cave? For whom amid the roses, many-hued, Do you bind back your tresses’ yellow wave? How oft will he deplore your fickle whim, And wonder at the storm and roughening deeps, Who now enjoys you, all in all to him, And dreams of you, whose only thoughts he keeps. Wretched are they to whom you seem so fair; That I escaped the storms, the gods be praised! My dripping garments, offered with a prayer, Stand as a tablet to the sea-god raised.",
        "author": "Roswell Martin Field",
        "date": 1891   },
{
        "text": "Pyrrha, what beauteous boy bedecked with roses, And dewed with liquid perfumes sweet and rare, In the sweet cave to you his love discloses? For whom do you array your golden hair! Simple its charms! But he– alas! How often Must moan your broken troth and Fate’s cruel doom, And wondering stare on seas that ne’er will soften From their wild rage beneath fierce heaven’s gloom! He now enjoys as gold your soft caresses, And dreams you are still faithful and still kind: He knows not of the treachery of your kisses, False as the fondling of the fickle wind. Wretched are they who trust to thy wild motion! But thankful to the Ruler of the Brine, I, who was wrecked upon this self-same ocean, Have hung my dripping garments at his shrine.",
        "author": "James A. Tucker",
        "date": 1891   },
{
        "text": "What dainty boy, with roses crowned, And steeped in perfume crystal clear; His arms thy lovely form around, Breathes love into thy list’ning ear? For whom, within the pleasant caves, Secluded in thy grassy bed, Bind up thy hair– those golden waves That cluster round thy lovely head? Alas! how oft will he bewail Thy troth, and Heaven’s baneful eye Will list; a novice to the gale, And to the tempest’s angry sigh. Who, pure in hope, basks in thy love, Nor dreams of thee as still unkind But unestranged– until he prove The wayward essence of thy mind.",
        "author": "D. M. MacR., Oban Times",
        "date": 1891   },
{
        "text": "What youth sighs in thy rose-heaped bowers, His locks perfumed with breath of flowers, For whom with simple grace and air, Does Pyrrha bind her golden hair? Alas! when o’er thy broken vows And Gods unkind, in grief he bows He sees not in that treacherous deep Rough waves to come, dark storms to sweep. Trustful he hopes these golden hours Will last, and always bloom these bowers; Changeful as air you but beguile, Unhappy they on whom you smile. I once a dismal wreck became, The sacred tablet bears my name, The powerful Sea-God hears my call, My garments grace his temple’s wall.",
        "author": "John B. Hague",
        "date": 1892   },
{
        "text": "What slender youth, sprinkled with perfumes rare, Courteth thee, Pyrrha, ’neath some rose-strewn grot? For whom dost thou unite the envious knot That binds thy sunny hair, Neat in thy comeliness? How oft shall he Changed luck, alas, and broken fate bewail, And waters ruffled by the dark-winged gale, Who now delights in thee? Too trustful boy! He deems thee golden all; Still fondly hoping he may ever find Thee all his own; as now, and ever kind, Nor reeks of treacherous squall. Unhappy they on whom thy splendours shine Untried! Yon wall and votive tablet show My dripping garments offered long ago Before the sea-god’s shrine.",
        "author": "H. Smith Wright",
        "date": 1892   },
{
        "text": "What stripling now his suit upon thee presses, drencht with his scent, and roses brow o’er-winding, on the cavern-floor? O Pyrrha, those blonde tresses for what man art thou binding, simple and sweet? What curses shall he heave at promises and shifty gods! What wonder shail fill his heart as dark winds strive to cleave the angry waves asunder! He thinks a lady golden-fair he has got, believes thee ever free and loving truly: thy treacherous gale he knows not. Sad their lot on whom thou shinest newly! A tablet shows within the sacred fane that there my dripping garments I suspended, a tribute to the God who rules the main where my life nearly ended.",
        "author": "William J. Ibbett",
        "date": 1892   },
{
        "text": "What slender youth with roses crowned, With liquid odors perfumed well, My charming Pyrrha, hast thou found To woo thee in his pleasant cell, For whom dost braid thy yellow hair And don thy simple robe with care? Alas! how often shall he weep For broken vows and gods estranged, Who, dreaming by the glassy deep, Beholds amazed its aspect changed, Black winds and surging waves arise For gentle airs and summer skies, Who now enjoys thy golden prime And hopes thou’lt always be his own, Loving and lovely all the time As if false winds had never blown. Ah, wretched they who win thy smiles And have not proved thy artful wiles. With me it is a thing gone by; In Neptune’s temple, on the wall, A votive tablet tells that I Have met with storms and baffled all, And hung my vestments dripping wet A sign,– where they are hanging yet.",
        "author": "John Osborne Sargent",
        "date": 1893   },
{
        "text": " What slender youth bedewed with perfumes sweet Woos thee on roses in a pleasant grot? For whom so simple, Pyrrha, and so neat Binds thou thy flaxen tresses in a knot? Alas! how oft will he with tears deplore Thy broken troth, the gods’ inconstancy, And wonder at the gloom unseen before Cast o’er the ruffled surface of thy sea! Thou, precious prize, at present art his own, No other swains, he hopes, his treasure share, Thy loving smiles are all for him alone, As yet of treacherous breezes unaware. Wretched, on whom thy love untried doth shine! A votive slab, and garments drenched with brine, Hung on the wall of Neptune’s temple, show To whose strong arm my safe escape I owe.",
        "author": "T. A. Walker",
        "date": 1893   },
{
        "text": "What slender youth ’mid many a rose, From whom a stream of fragrance flows, In cool grot, Pyrrha, wooes thee? For whom dost twine thy golden hair So trimly artless? Broken troth He oft will mourn and gods grown wroth, And waves ’neath tempests curling He’ll view aghast in his despair, Who now, bewitched by golden wiles And bent on hope of unshared smiles For ever, fickle breezes Suspects not. Hapless they will be On whom thy glamour untried glows. Yon temple’s votive tablet shows That I with dripping raiment Have dowered the god who rules the sea.",
        "author": "J. Howard Deazeley",
        "date": 1894   },
{
        "text": "What graceful youth, O Pyrrha, found In odours steeped, with roses crowned Thy fancy now hath caught? For whom, with simple charm arrayed Dost thou thy yellow tresses braid In some delightful grot? Alas, how oft will he bewail Thy shifty faith– the sudden gale The darkling waters rue! Though you are golden, he is blind, Trusting that you’ll be always kind, That you’ll be always true! Oh, wretched they for whom you shine A sea untried! The votive shrine Is now the place for me; Lost on the waves no longer now, My garments dripping brine I vow To Him that rules the sea!",
        "author": "W., Ronald Storrs Papers",
        "date": 1894   },
{
        "text": "What youth so elegant, faint with perfumery, Presses thee tenderly, now, on the roses, In the cool shade of thy grotto, O Pyrrha? And now, for whom dost thou bind up thy yellow hair Simple and daintily? Ah, but how often yet He shall sigh bitterly over thy faithlessness, And fickle deities! He who delights in thee, Thinking thou’lt ever be placidly beautiful: What will his wonder be when the rough seas arise, Dark with the hurricane! When he now, credulous, Deems thou wilt ever be gentle and lovable; Of the winds treacherous, he is unmindful. Woe to the novice, who finds thee alluring! My votive tablet there,– high on the temple wall, Tells I have gratefully hung my wet garments up, To the god consecrate, ruling that Sea!",
        "author": "George M. Davie",
        "date": 1894   },
{
        "text": "What graceful, perfumed youth on many a rose, ’Neath pleasant grotto, doth with thee repose, Pyrrha? For whom thy tawny hair Dost bind– so simply fair In thy adornment? Ah! how oft shall he Weep his changed gods and faith, and at the sea Wonder– which, all unused, he finds Ruffled with angry winds Who, fond, enjoys the golden prize of thee, And hopes thee ever loving, ever free To have– of breeze deceitful aye Unweeting. Wretched they For whom thou shin’st untried! Yon tablet set On sacred wall proclaims that garments wet I have suspended there to please The god that rules the seas.",
        "author": "William P. Trent",
        "date": 1894   },
{
        "text": "What nice young man with perfumed hair And button-holed with florets rare, Under the arch in the open air Thee, Pyrrha, strives to please? No doubt he thinks your tidy trim Is all got up to capture him: But soon he’ll have to sink or swim When change the gods and seas. Poor fool! Who thinks he’s got you fast, That pleasure, leisure, love will last, Nor hears a whisper on the blast Which soon his hopes shall strand. O luckless wights that trust to thee, As to a smiling, untried sea; One bare escape’s enough for me; Thank God! I’m safe on land.",
        "author": "Anonymous, Pall Mall Gazette",
        "date": 1895   },
{
        "text": "What slender youth with liquid perfumes fair, Pyrrha, amid lavish roses woos thee? For whom now bindest thou thine amber hair, Adorned yet unadorned simplicity, In some cool grot? How soon, alas! will he Thy broken troth and slighted pledge deplore, And in amaze will gaze upon the sea Blackening ’neath stormy winds and calm no more. Now basks he fondly in thy golden smile, Trusting that loyal thou wilt ever prove, And loving ever; knows he not the guile Of woman’s heart, the shifting breeze of Love? Sudden thou smilest on him! sad his fate! But I, from shipwreck ’scaped, to Ocean’s king A votive tablet duly dedicate, And dripping garments as thank-offering bring.",
        "author": "Cyril E. F. Starkey",
        "date": 1895   },
{
        "text": "What handsome boy, besprayed with liquid odours, Pyrrha, now courts thee in some pleasant grot, ’Mid many a rose? For whom, with simple grace Is twined thy golden hair? How oft, alas! Will he bemoan bad faith and gods estranged, And innocently wonder at the seas Rough with a darkling storm; who still confiding, Enjoys thy golden calm, and puts his trust In thee, imagined always kind and true: So little dreams he of the treacherous blast! Ah! wretched those, for whom thou glitterest yet, Untried! But as for me, the sacred fane Shows by the votive picture on the wall That I have hung my dripping garments there An offering to the ocean’s mighty god!",
        "author": "Charles Newton-Robinson",
        "date": 1895   },
{
        "text": "Say, what beautiful youth, with body scent-bedewed, On your flowery couch, Pyrrha, caresses you ’Neath that grotto delightful? Decked thus simply for whom do you Braid your golden-hued hair? oft, alas! he’ll bewail Troth and destiny changed, and alas! oft upon Dark storm-winds of a rough sea Gaze with wonder, unused to them, Who now trustful adores beauty so glittering, And, not knowing a breeze oft to be fickle, trusts Always gentle to find you, Always fancy-free. Woe to those Whom your charms unaware dazzle! The temple-wall With slab votive adorned is the memorial, That drenched clothes to the Sea-god’s Pow’r I’ve hung as an offering.",
        "author": "Oswald A. Smith",
        "date": 1895   },
{
        "text": "What slender youth with liquid scents bedewed Is courting you on roses thickly strewed, Pyrrha, in pleasant grot? For whom twist you that golden hair in knot So charming-simple? Ah! how oft he’ll weep For heaven’s changed looks, the troth you would not keep, And wonder, slow to learn, How rough in murky winds Love’s sea can turn. Now, lapped in golden joys, he fondly sees You always pleasing, always free to please; Poor fool! he little knows The fickle breeze that now so softly blows. The wretch is lost on whom you smile untried; My votive tablet on that wall, inside The mighty Sea-God’s shrine, Shows where I’ve hung my garments dripping brine.",
        "author": "A. S. Aglen",
        "date": 1896   },
{
        "text": "What silly boy perfumed with liquid unguents, Within thy sweet retreat, ’mid many roses, Urges his suit, O Pyrrha! For whom dost dress thy yellow hair So simply beautiful? Oft, alas! thy faith And changing gods unkind shall he bemoan, And gaze in wonder on the seas Rough with black winds tempestuous, Who now enjoys thee, thinking thee all golden, And always gay, with welcome to his wooing Hopes thee– change unsuspecting. O most unhappy are those youths To whom thou, unessayed, art fair. But for me, The sacred wall, by votive tablet, shows my robes Still dripping, there suspended To the powerful God of Sea.",
        "author": "John G. Freeze",
        "date": 1896   },
{
        "text": "That dainty youth, bedewed with sweet perfume, Pyrrha, caresses thee, where roses bloom Beneath thy pleasant bower? For whom dost thou now bind thy golden hair In neat simplicity and fashion rare, To give thee witching power? How often shall he, fool that he must be, Wonder at changed fates, at troubled sea, And weep thy broken faith? He who all precious thee does now enjoy; Who deems thee fancy free, without alloy, Poor blinded, storm tossed wraith. Wretched are those whom, knowing not thy wiles, Thy beauty and simplicity beguiles, To shipwreck make of each. My dripping garments, hung in Neptune’s shrine, For my escape vowed to the god divine, May they a lesson teach.",
        "author": "Richard Fenwick Ely",
        "date": 1896   },
{
        "text": "What youth, slender and fair, ’mid the thick rosebushes, All with odours besprent, Pyrrha, caresses thee ’Neath thy grotto delightful? For whom bind’st thou thy golden hair, In simplicity neat? Ah me! how oft shall he Mourn thy perfidy, and skies, that were once so fair, Clouded over with tempests And the change of the Gods to him, Who enjoys thee, poor fool! now in thy golden hour! Hoping thou wilt be all-loving, all fancy-free, Knowing not the deceitful Breeze. O! wretched are they to whom All untried thou art fair! Me! Neptune’s holy wall Shews, on tablet devote, to have hung up aloft Garments dripping with Ocean, Sacred to the strong God of sea!",
        "author": "Philip E. Phelps",
        "date": 1897  },
{
        "text": "What slender youth, on scattered roses lying, Woos thee, fair Pyrrha, in some cool sequestered place? For whom bind’st thou thy yellow hair With artless grace? Ah, hapless boy! how soon, how soon to tears Will his young golden dream be turned, when clouds arise On that bright sea, and changed gods Avert their eyes! Who now has all thy love, nor dreams that thou Could’st change, could’st ever cease to love him, or the day Could come when love and faith would fail Ah, wretched they, For whom thy beauty shines! My dripping weeds, Hung on great Neptune’s votive wall, proclaim for me To all, how I erewhile escaped That cruel sea.",
        "author": "Edward George Harman",
        "date": 1897  },
{
        "text": "Pyrrha! what slender youth in perfumes steeped courts thee ’mid circling roses in thy pleasant bower? for whom dost bind thy yellow locks with simple grace? Alas, how oft shall he weep his outraged troth, his fortune changed, and stand amazed at the waves that rise before the blackening squall– poor credulous novice, who dreams thou wilt ever be his alone and meet for love, all ignorant of thy favour’s fickle breeze! Hapless they who see thy beauty and know thee not! But I, as yon temple wall’s votive tablet declares, have hung up my dripping raiment as a thank-offering to the god who rules the main.",
        "author": "Alfred Denis Godley",
        "date": 1897  },
{
        "text": "What spruce young lad with roses garlanded, Distilling sweets from Araby the Blest, Courts thee, my Pyrrha, in a grot’s kind shade? For whom hast thou thine amber tresses drest So simply sweet? Ah me, how oft will he Make moan on broken troth and gods averse. And marvel helpless at the storm-tost sea And fickle gales, untutored of their course, Who now, too credulous, thinks thee pure gold, Hopes thou wilt ever welcome him, wilt love And be beloved, not versed like me of old To read the moods that flit thy brows above. Ah, hapless wights, who trust thy treacherous face! The chapel wall; with painted shipwreck, tells How there I gave my dripping garments place To Neptune hallowed, victim of thy spells.",
        "author": "W. A. Heidel",
        "date": 1899  },
{
        "text": "Pyrrha, pray, what perfumed stripling woos thee now? For whom, beneath thy rosied grotto, dost thou bind Thy golden hair, in artful sweet simplicity? Alas, full frequent will he weep, dear innocent, His former faith and changeful gods, and marvel how The cruel waters rage, ’neath black and wrathful winds. Thy charms he now enjoys, and dreams thou art all gold, And ever free of heart and ever kind, he hopes, Nor knows that fitful breeze, thy fancy’s fickleness. Unhappy they on whom thy bright eyes gleam untried The sea I’ve fled, and to Poseidon consecrate On temple wall these lines and garments stained by storm.",
        "author": "Paul R. Wright",
        "date": 1899  },
{
        "text": "What slender stripling, charming girl, Woos thee in rosy bowers, Who wantons with thy golden curl Mid beds of lovely flowers? For whom dost bind thy golden hair, Who thinks thee ever true? Thou tempting yet delusive snare, Alas! This day he’ll rue The winds will change, the youth will find That tender chords not always bind, The sea that now so gently glides, Will soon be all o’ercast with tides Oft will he deprecate his fate, Repent,– but ’tis one day too late, With tears deplore her heart untrue, This glittering jewel loves the new These lines I consecrate on land, I too was ship-wrecked on that strand.",
        "author": "W. K. G., The University of Virginia Magazine",
        "date": 1900  },
{
        "text": "What graceful youth, with roses crowned, Bedewed with liquid perfumes rare In some sweet grotto’s sacred ground, Pours forth to thee his ardent pray’r? For whom dost thou thy golden hair In simple tresses careless bind? Alas! how oft shall his despair Blame thy false faith, and gods unkind? Whose young and inexperienced mind, Now awed by storms that lash the main, Now fondly credulous and blind, And, too confiding, trusts in vain, That from all other fondness free, Thou true to him shalt ever be. Ah! wretched they to whom, as yet Untried, thou seemest fair! This votive tablet which I’ve set Against the wall of Neptune’s fane, Tells how my garments, dripping wet, I’ve vowed to him who rules the main.",
        "author": "C. G., Dublin Weekly Nation",
        "date": 1900  },
{
        "text": "What tender youth with liquid scents perfumed, In pleasing grot, with many a rose adorned, Pyrrha, to thee makes his impassioned suit? For whom, with plain and simple elegance Thy golden tresses braidest thou? Alas! How oft shall he bewail thy faith unkept, His gods estranged, and, to the sight unused, Wonder at seas with blackening tempests rough, Who, foolish now, delights in thee as pure, True gold; who, ignorant of fickle gales Thee always vacant hopes and always kind. Unhappy they to whom thou yet unproved Art lovely! Me the sacred wall proclaims, By votive tablet, to the god that rules The sea my dripping garments to have hung.",
        "author": "Benjamin F. Meyers",
        "date": 1901 },
{
        "text": "Pyrrha, what slender youth, with perfumes moist bestrewn, Doth midst a bank of roses claim thee for his own Beneath some cavern roof? Thy yellow hair is tied for whose behoof? Such plainness marks thy beauty strong; and oft with pain, Thy lovers moan for broken troths, for worships vain: When thy wrath meets their gaze, Its buffets cruel strike with sore amaze. Who trusting now thinks thee all-perfect as fine gold, And deems in undisputed sway thy charms to hold, Ignores the coquette’s whim; Deceiving others, she deceiveth him. How fares the wretch who, on thy favors, would repose Yon sacred wall with painted, votive offering shows: My drippling garments see Hung up for Neptune; I am saved from thee.",
        "author": "Thomas Ewing Smiley",
        "date": 1902 },
{
        "text": "What slim boy woos in pleasant grot With liquid odours round him shed Thee, Pyrrha, on rosy bed? For whom thy yellow hair dost knot, Plain in thy daintiness? Ah me! How oft lost faith and gods that change He’ll weep, with wonder strange At the black winds that fret the sea! Who credulous now thinks thee gold, Hopes thee still dear, still all his own; Thy changeful breeze unknown, How false. Poor dupes, who real hold Such untried sheen! I, as the board limned on the temple wall doth show, My wet weeds long ago Hung up in thanks to ocean’s lord.",
        "author": "W. C. Green",
        "date": 1903 },
{
        "text": "What dainty youth, O Pyrrha fair, Bedewed with liquid perfumes rare, Embraces you in pleasant grot, In some sweet, rose embowered spot? For whom dost thou thy locks of gold So simply bind, so neatly fold? Alas! How oft shall he deplore Thy perfidy and gods no more Auspicious! Inexperienced, he Shall wonder greatly at the sea, So roughened by the blackening storms, Who now, in thee believing, warms His heart with thee, O precious fair; Who, ignorant of the faithless air, Now hopes to find you always free, Always good-natured as can be. O wretched ones, to whom, untried, Thou seemest fair! The sacred side Of Neptune’s temple shows to all By votive tablet,– not too small That I my wet clothes consecrate To the sea-god of powerful state.",
        "author": "R. C. D., The Round Table",
        "date": 1903 },
{
        "text": "What slender young stripling, amidst many roses, Be-perfumed with odorous liquids, pursues thee, O Pyrrha; in grotto delightful, For whom thy fair tresses entwined are With art of the simplest? Alas, how oft faith, and Mutations of gods– when his erstwhile calm ocean Is roughened by blackest of storm winds Amazed, he’ll deplore soon! Who, credulous, precious as gold now enjoys thee, And aye free and faithful believes, in his hope, too; Ignoring thus ever the tempests Deceitful. Ah, wretched are those who Untried hold thee fair! Lo, my own votive tablet The sacred-wall shows, with my still-dripping off’ring Suspended beneath it to him that is potent My robes– to the Sea-god!",
        "author": "Clarence Cary",
        "date": 1904 },
{
        "text": "What slender youth, aloof, where dew-stared roses shed Their perfumed balm within thy grotto, woos thee hence, O Pyrrha, of the golden braided head, And studied innocence? Alas, how oft, lost faith; the frown of changing gods, Shall he deplore, when his erst placid, favoring sea, May, whilst he wantons dreamingly, and nods, Darkened with tempests be! Thy dross, he, witless, holds all golden in emprise, Nor heeds the brooding storm, nor idle hope gives o’er, That all the wondrous love-lights of thine eyes Are his– for evermore. Thus hapless, who, untried, believe thy worth. But lo! My votive tablet on yon temple-wall shines fair To ocean’s god, who, from wrecked overthrow, Saves mortals, here and there.",
        "author": "Clarence Cary",
        "date": 1904 },
{
        "text": "What slender youth with odours dewed is he Courts thee within the pleasant cave reclined, Pyrrha, with many a rose? For whom dost bind Thy golden hair in the artless subtilty Of thy adornings? Oh, how oft on thee Shall he complain, and the changed gods unkind And waves, unwonted, rough with blackening wind, Who now all credulous of thy golden sea, Thus always vacant, always amiable, Hopes thee, of the fair-seeming gales untried. Ah, hapless they, on whom has shined thy tide Yet unessayed! For me the wall may tell In my vowed picture to have bestowed in fee My dropping weeds to Ocean’s deity.",
        "author": "M. Jourdain",
        "date": 1904 },
{
        "text": "In thy pleasant grotto, Pyrrha Dense with odors of the forest, And the fragrance of fresh roses, Who upon thy breast reposes? Who this last youth thou adorest? Deck thy dewy locks, O damsel, For this latest, lealest lover, With the amaranth and laurel; While these last, may lovers’ quarrel Ne’er thy heartlessness discover! While the wave the wild wind tosses, And the white crests crown the billow, Do not with his young heart trifle, Nor thine own affection stifle; Yield thy breast a willing pillow. So he’ll never know how changeful Is thy love, prone to transition; How, if one have ceased to please thee, Other lover can appease thee With a new love’s sweet fruition. In the temple of the sea-god, Down beside the surging ocean, I have hung my wet apparel; From this time, O maiden, farewell! Neptune claims my heart’s devotion.",
        "author": "Fabius Maximus Ray",
        "date": 1904 },
{
        "text": "What slender youth with flaxen hair Is he that now thou smilest on, O Pyrrha? Are his senses gone, To hope thy mood is always fair? My faith! When o’er thy lips and eyes The lightning smile of mirthful scorn, That comes so soon, is swiftly born, He’ll think the sun has left the skies. He has my pity, hapless youth, Who knows thee not as other men; Alas, poor lad! He’ll learn and then He’ll see the gulf ’twixt thee and truth. I know thee! I have saved my neck From death by drowning in love’s deep, And smiling Neptune long will keep His memory of my stranded wreck.",
        "author": "William J. Henderson",
        "date": 1905 },
{
        "text": "What graceful youth, with roses crowned, And scented locks, by soft caress Controls thee, girl; for him unbound In shady grot, each auburn tress? Simple in elegance! alas! How oft at changing skies he’ll rave And, unaccustomed, watch as pass The darkening clouds, and surging wave; Who now, in perfect faith, thy loves As golden deems; as his alone; And ever amiable: which proves The treacherous breeze to him unknown. Unhappy those, on whom, untried, Thou shinest– for me, the sea-god’s fane, As votive offering, shows, half dried, My garments hung; to these remain.",
        "author": "Eccleston Du Faur",
        "date": 1906 },
{
        "text": "Amid many a rose, what slender youth bedrenched With liquid odours, wooes you, Pyrrha, Within a pleasant grot? For whom Do you braid your yellow hair with art That apes simplicity? Ah fie! how oft Shall he bewail your faith, and changed gods, And upon seas enruffled with Black winds astonied gaze? Who now enjoys you credulous all gold, Whose hope conceives you ever free and fond, In ignorance of your fickle airs! Misfortunate they on whom, Untried, you shine! A sacred wall By votive tablet indicates that I Have hung wet garments to the god Who dominates the sea.",
        "author": "Edward R. Garnsey",
        "date": 1907 },
{
        "text": "What slender youth, with wealth of roses sheen And with sweet essences besprent, pursues thee, In cool grot, Pyrrha, woos thee? For whom thy yellow hair dost preen, Simple yet exquisite? How oft, ah me! Vows broken he’ll deplore and gods that change; And, to thy whimsies strange, Shall gaze where glooms a wind-swept sea; Who credulous now dotes on thy tinsel gold, And dreams thee ever willing ever kind, To thy fair falseness blind! O hapless, who untried behold Thy glitter! Lo, my dripping weeds I place, With picture vowed, on Neptune’s temple wall, My saving to recall From shipwreck by thy siren face.",
        "author": "John Marshall",
        "date": 1907 },
{
        "text": "What slender youth, I pray, with odors sweet bedewed, On mossy, shady bank with many roses strewed, Now woos thee, Pyrrha, and caresses? For whom braid’st thou those shining tresses, In ornaments so neat? How oft will he complain Of broken faith and cruel gods, at stormy main And black clouds sweeping through the skies Unwonted gaze with sad surprise, Who fondly trusts thee now while sunning in thy smiles, Imagines thee pure gold, unused to maidens’ wiles, Hopes winds will always softly blow. That bring, alas, but frost and snow To unsuspecting wights who on thy glances hung! I, on the sacred wall, my vestments drenched have flung. By pictured tablet raised to thee, O mighty ruler of the sea!",
        "author": "Eugene Parsons",
        "date": 1907 },
{
        "text": "What youth, with many roses, slim, With liquid odours sweet, o’er him, Woos thee, Pyrrha, beneath some rock’s Cool grot? For whom bind’st thou thy locks; Thou simply neat? Ah! oft, trust lost And changed gods he’ll weep, and tossed Waves, stirred by darkling winds, t’ his eyes All new, will hold his long surprise, Who, now thee golden deeming, thee Enjoys, who hopes thee always free, E’er lovable– he mindful not O’ the treacherous gale. To those hard lot, Thou shinest on new. The sacred wall On votive tablet doth install My garments wet, as hung by me, Tithes to the god, who rules the sea.",
        "author": "J. A. Salmon-Maclean",
        "date": 1907 },
{
        "text": "Who the slender stripling, Pyrrha, that woos thee now, Wreathed profuse with roses under the rock’s kind brow, Fragrant of dewy odours, whilst thou, for his delight, Decked in a dainty neatness, dost weave thy tresses bright? Ah! how oft thy faithless faith and the gods unkind He shall weep, and, wondering, gaze at the angry wind Blackening the ruffled waters, whom now thy golden grace Lulls in a sweet delusion to dream thy winsome face Ever free and fond; unwarned of the changing tide. Hapless they on whom thou glitterest, yet untried! I, on the votive tablet beside the sacred shrine, Tell I have hung dank garments to him that rules the brine.",
        "author": "Sarotti, T. C. D. Miscellany",
        "date": 1908 },
{
        "text": "Who in the light of your beauty reposes, Drenched with the odors of flowers that are sweet? Who is the swain in your garden of roses, Breathing his vows at your exquisite feet? Weaving your passionate spells unbeholden; Twining each band for a glittering snare; Daintily coiled and deliciously golden; Pyrrha, for whom have you braided your hair? Ah! for the lover whose trusting devotion Hears not the far-away tempests that roar, Heeds not the wrath of the terrible ocean, Sailing so smoothly away from the shore. Ah! for the shallows where brave ships have foundered, Washed by the white waters seething above. Ah! for the depths where the doomed ones have foundered, Under the waves of your pitiless love. I, on the beach where the billows have tossed me, Sorrowing still for the treasures they keep, Wrecked in the ruin of dreams they have cost me, Pity those others who steer for the deep.",
        "author": "Anonymous, The Lone Hand",
        "date": 1908 },
{
        "text": "Ah, Pyrrha, amid the roses What perfumed gallant now His heart to thee discloses ’Neath some cool grotto’s brow? For whom dost thou Now braid thy tresses golden With artless grace? Alack, Though now clear skies embolden, Soon shall he quail ’neath black Storm-clouds and wrack, Who trusts, poor fool, thy favour, Nor dreams that storms can rise, Or anger darken ever The heaven of thine eyes. Hapless the unwise On whom thy smiles are squandered! But I for perils o’er Thank-offering meet have rendered, And tempt the seas no more, Saved and ashore!",
        "author": "A. Margaret Ramsay",
        "date": 1909 },
{
        "text": "What slender boy with curling, perfumed hair ’Mid clustering roses, woos you, Pyrrha fair, In your sweet bower? Tell me for whose delight Are bound and bound again those tresses bright? How seeming simple is your daintiness! Yet soon, alas! he’ll mourn your fickleness, The gods estranged. As when the sailor pale Sees billows roughen ’neath the black’ning gale Where all was fair; so he, unconscious, blind, Believes you always loving, always kind; Enjoys your smiles nor fears the changing wind. Poor hapless fools, for whom, untried, you shine! Well, let them learn their lesson,– but for mine, My sea-soaked garments, with a thankful prayer, I’ll dedicate to Neptune for his care.",
        "author": "Elizabeth H. Du Bois",
        "date": 1909 },
{
        "text": "Pyrrha, what stripling woos your loveliness? What gallant brings his perfumed grace to press Your roses’ sweetness? For what new victim do you deck your lair, Pyrrha? For whom bind back that golden hair; In witching neatness? How oft, alas, shall he, poor novice, find With weeping eyes nor plighted troth shall bind Nor Gods shall harken, When he– how little skilled or weather-wise Amazed shall watch the angry waters rise And heavens darken; Who now enjoys you on a golden sea, Who dreaming you still fond, still fancy-free, Now basks beside you. I thank the Gods I swam ashore: time was I sailed like him a trusting fool– because I had not tried you.",
        "author": "Harold Baily Dixon",
        "date": 1910 },
{
        "text": "What dainty stripling scent bedewed Woos thee on bank with roses strewed, Pyrrha, ’neath pleasant grot? For whom dost bind thy yellow hair In unbedizened neatness fair? How oft the promise kept no more, And Gods estranged shall he deplore, And in unwonted lot Watch wondering the black squalls that sweep To angry surge the roughened deep; Who revels in thy golden charm, Too credulous, nor knows the harm Of winds that wax and wane; Who hopes thee evermore to find Obedient to his beck and kind. Ill-starred for whom thy beauty glows Untried: but I, the tablet shows Nailed to the holy fane, Have hung my dripping weeds on high A gift to Ocean’s deity.",
        "author": "Francis Law Latham",
        "date": 1910 },
{
        "text": "What slender boy besprent with wet perfume Pleads with thee, Pyrrha, on a rosy bed In pleasant bower? For whom That yellow tress upon thy head, Thine artless art? Alas, how oft in vain Shall be thy troth and gods forsworn lament, Gazing in wonderment Upon that dark, wind-ruffled main, Who now, poor fool, enjoys thy golden youth, And hopes thou’lt aye be constant, aye be kind, Nor wots of fickle wind. Alack for those who trust thy truth And see thee glittering fair! On holy shrine A votive pannel vouches that by me My vesture wet with brine Hangs offered to the lord of sea.",
        "author": "J. H. Hallard",
        "date": 1910 },
{
        "text": " What slim youth now, bedewed with soft perfume, On banks of roses thee caresses, O Pyrrha, hid in some cool cavern’s gloom? For whom dost bind thy golden tresses In graceful neatness? Ah, how oft will he His misplaced confidence bewail, Who, inexperienced, wonders at the sea Aroused and darkened by the gale! Yet thou as gold delectable dost seem To his too easily bedazzled eyes, Who thinks thee ever true, without a dream That storms may take him by surprise. Unfortunates, to whom thou like a sea Untried, dost yet alluring shine! A tablet hung on Neptune’s wall by me Shows what a shipwreck late was mine.",
        "author": "George M. and George F. Whicher",
        "date": 1911 },
{
        "text": "What slight youth at thy feet, Pyrrha, doth sue, I crave, Bathed in odors so sweet, under a rose-strewn cave? Pray, for whom is that twining, Simple fair, of thy golden hair? Ah, how oft shall he weep over thy lealty changed, Oft with shudderings deep over the gods estranged, And, unwonted, repining, Startled be at the wind-tossed sea. Golden pure dost thou seem. Trusting, he joys in thee. Thou, he fondly doth dream, e’er for his love art free, Not the fickle wind knowing, Wretched they in whose luckless way. Thou, unproven, dost shine. Now from the sacred wall This vowed tablet of mine, my dripping garments all, Wreck-delivered, is showing, Consecrate to the sea god great.",
        "author": "Myra Pollard",
        "date": 1911 },
{
        "text": "What graceful youth, bedewed with sweet perfume, Caresses thee on beds of rosy bloom, Within this grotto fair? For whom thine auburn hair Dost thou in simply knotted neatness groom? How often shall he weep the gods aloud, And stand amazed before their thunder cloud, Who now enjoys thee, gold! He prays that, as of old, Thou wilt be free– unsought for by the crowd. Wretched are they for whom thou shinest, untried. Come, as a goddess who hast long defied My tender offerings, Ease thou my sufferings, Accept my votive, I have braved thy tide.",
        "author": "Howard James Savage",
        "date": 1911 },
{
        "text": "What scented stripling woos thee lying, Pyrrha, in grotto fair, ’Mid many a rose? for whom art trying Thy auburn hair With simple grace? Poor boy, how often Thine and the gods’ caprice Shall wilder him, like squalls that roughen His sunny seas! He thinks thee gold, he hopes that ever Thou wilt be free and kind, Nor dreams of veering winds. Ah, never Were folk as blind As they who’ve proved thee not! my payment In yonder fane is stored: A tablet vows my dripping raiment To Ocean’s lord.",
        "author": "William S. Marris",
        "date": 1912 },
{
        "text": "What scented slender youth ’mid roses fair In pleasant grotto, Pyrrha, courts thee now? For whom that red-gold hair Dost simply bind from brow With graceful art? Ah, often will he weep O’er faith and gods that change, and, new as yet, Marvel when placid deep Black winds to roughness fret. Too trusting, now as gold he joys in thee, Hopes thou wilt ever be heart-free and sweet, Nor knows how breezes be Inconstant. Woe who meet Thy bright gleams, thee untried! For me, all wet My garments I have hung to God who sways The sea; the tablet set In temple shows my praise!",
        "author": "J. M. Krause",
        "date": 1912 },
{
        "text": "Who is that handsome boy, O Pyrra fair, With perfumed locks who, seeking, thee discloses, And, fervent, woos thee of the golden hair In yonder grot, entwined with blushing roses? And dost thou bind the yellow tress for him In simple neatness, giving him no notion That he will weep thy ever changeful whim Which breaks, like squalls upon the summer ocean? He now enjoys, fond fool, thy glorious youth, He hopes thee ever leisured, ever gracious, Nor knows, unconscious, that thy seeming truth Is, like the fickle sea and wind, fallacious. Unhappy victims of thy smiling guile! Of such am I; my votive gift suspended, Marks my escape from shipwreck on Love’s isle, Whose altar takes my sodden clothes, unmended.",
        "author": "J. Wells Thatcher",
        "date": 1913},
{
        "text": "What graceful youth now woos thee there In pleasing grotto, Pyrrha fair, In liquid perfumes steeped His hands with roses heaped? For whom dost thou so seeming-kind Thy golden locks in beauty bind, Holding him in amaze With all thy charming ways? Alas, how often shall he mourn The fickle gods, the faith forsworn, And all astonied be To mark the wind-vexed sea! Who now enjoys the golden hour Nor thinks, poor wretch, of clouds that lower, Of door that opens not, Thy love now quite forgot. Hapless are those who gaze on thee As men upon a shining sea, And credulously dare The untried waters fair. But I, escaped those billows rude, Do thus with solemn gratitude That voyaging recall, And on the sacred wall The votive tablet now have set That tells of garments dripping wet Hung in his shrine whose sway All ocean’s waves obey.",
        "author": "A. L. Taylor",
        "date": 1914 },
{
        "text": "What slender stripling all adorned with roses, And hair oil dripping from his odorous locks, Woos thee, O Pyrrha, and his love discloses, ’Neath the cool shadows of o’erhanging rocks? For whom, pray, dost thou bang thy yellow tresses In simple swirls? Alas! How soon he’ll learn Capricious Cupid’s and thy false caresses, Astonished somewhat, when thou callest the turn. At surging seas and at night so pitchy, In present bliss he thinks thee solid gold And free of flirting. Mercy! ’tis this which he Hopes, knowing not that he is badly sold. Unhappy they on whom thou sweetly smilest! The bath house walls bear witness now for me; I’ve hung my dripping shirt from bath the vilest, A tribute to the monarch of the sea.",
        "author": "Harvey W. Wiley",
        "date": 1914 },
{
        "text": "What chap is this, so slim and young, Who smells of perfume and is hung With roses; who makes love to you In shady corners just for two? Unfilleted your yellow hair, Yet lots of folks will up and swear At your false words and fickleness: And he who now enjoys your Yes And thinks you always gay and sweet He’ll find out soon that your deceit Is like soft airs that blow, but soon Develop into a typhoon Disclosing depths of waters dark Where navigating is no lark! Myself, I’ve rained a tablet to The ocean god who pulled me through. And there, close to my dripping cape, It tells to all my narrow escape.",
        "author": "Edith, The Conning Tower",
        "date": 1914 },
{
        "text": "What stripling boy, with fragrant dews besprent, Clasps thee mid many a rose in pleasant grot? For whom, O Pyrrha, art thou bent Thy yellow tresses now to knot In studied artlessness? How oft, alack! Will he deplore changed faith and gods untrue, And, while downswoop the tempests black, The roughened seas appalled will view, Who now, bewitched beneath thy golden spell, Hopes thee for aye his own, lovely for aye, Unweeting of the stormwind fell So soon to blow! Most wretched they Who trust, unproved, thy dazzling loveliness! I know; yon sacred wall my picture keeps In witness that my brine-soaked dress Is vowed to Him who rules the deeps.",
        "author": "Warren H. Cudworth",
        "date": 1917 },
{
        "text": "What dainty youth with perfumes rare bedewed Caresses thee within that rose-sweet cave, O Pyrrha! for whom now thou Dost bind back thy yellow hair In girlish neatness? Alas how many times He will bemoan thy perfidy, and since Innocent, wonder at seas Wind black and rough with storms, Who loving well thy glad gold moods, dost fondly Dream thee ever kind and disengaged, Naught knowing of treachery. Oh! wretched, indeed, are they To whom untried thou shinest fair afar! Now I on Neptune’s silent temple wall My out-worn garments have hung up To honor the Sea God.",
        "author": "Edna Worthley Underwood",
        "date": 1917 },
{
        "text": "I ask you, Pyrrha, is there any slender boy, Perfumed with sweetest waters, who can still enjoy To wreath your head with roses in some sheltered grot, That you so bind your flaxen tresses in a knot? Alas, how oft will he lament your faith untrue, And fickle Destiny! Alas, how he will view And wonder at the sea deep-stirred by every breeze, For that he did not know that maids love as maids please! Alas for him, too trusting in your fickle love, Who holds you as a priceless blessing from above; Who, ignorant how the fleeting storm has waged, Expects at all times he will find you disengaged! Alas for him who knows you not, so deems you fair! That is a garment I have ceased for long to wear. A votive tablet on the sacred wall shall be A sign I offer it, a tribute, to the sea.",
        "author": "J. Carey Thomas",
        "date": 1917 },
{
        "text": " What dainty, perfume-scented youth, whenever he proposes, Caresses you, oh Pyrrha, in a pleasant grot and fair; For whom do you reveal your charms among a thousand roses? For whom do you bedew your eyes and bind your shining hair? Alas, how soon shall he deplore your perfidy, when lonely He shall behold the altered gods, invisible to us, Who now believes you his alone and who enjoys you only, Who hopes (so credulous is he) things will be always thus. Oh woe to those, the luckless ones, who cling to you, not knowing Your faithlessness and folly– and to whom you seem so fair. Lo, on the wall of Neptune’s temple is a tablet showing My votive offering tendered to the Sea-God with a prayer.",
        "author": "Louis Untermeyer",
        "date": 1919 },
{
        "text": "Pyrrha, what slim young lad, in perfume bathed, Woos thee on roses in some shady grot? For whom with careful carelessness is swathed Thy yellow hair beneath the fillet’s knot? Thou art his golden dream, unmarred by fear Lest in thine heart his throne he fail to keep; Alas, unwarned how many times may veer The fickle breeze, how often shall he weep False faith and altered fortune, and shall stare With unaccustomed eyes on surges blown By the black wind! Unhappily they fare Who seek thy brilliance ere thy flame is known. But as for me, behold the neighbouring shrine, Where, on the mural tablet, I record How there I hung my garments, drenched with brine And dedicate to ocean’s mighty lord.",
        "author": "Francis Coutts, Lord Latymer",
        "date": 1920 },
{
        "text": "What scent-besprinkled stripling lad, Pyrrha, would win your favour, where Some grotto smiles with roses clad? For whom bind you your golden hair, Simple, yet dainty? Soon he’ll weep, How oft! changed troth, changed deities, And marvel, as the wind-lashed deep Darkens, and threats his startled eyes, Who in his folly counts you now All gold, and hopes that free for aye And kind you’ll be, unwitting how Your favours cheat. Unhappy they On whom you smile untried. For me, His temple-wall and tablet show That to the God, who rules the sea, I hung my drenched robes long ago.",
        "author": "William Hathorn Mills",
        "date": 1920 },
{
        "text": "In your rose garden, Pyrrha, false and fair, I wonder who– you’re not alone, I swear It is for whom within the kindly grot You bind, unbind, that wealth of sunny hair. Artless elegance! Can there be alarms, For any folded in those clinging arms, Of knitted brows, sharp words, and stormy eyes, Where late were none but smiles, and honeyed charms? Unknowing he how fast a wind can turn, And golden girls, as skies, now freeze, now burn, Deny their door, once never closed to him Poor innocent! how much he has to learn! Faithful for aye, he thinks. And I thought so, Ere I was cast adrift. Envy? Oh no! Joy to have ’scaped with life a Siren’s wiles. See, my clothes drip; ’twas not so long ago.",
        "author": "William Stebbing",
        "date": 1920 },
{
        "text": "Pyrrha, what stripling slim, in use Of perfumed essences profuse, Wooes thee on roses laid In some cool grotto’s shade? Say for whose eye thou hast designed, That knot thy auburn hair to bind And curb each errant tress, Thy neat but simple dress, How oft will he for vows forsworn And heaven’s regard averted mourn, How often with amaze On squally sea will gaze! Whom now, fond dupe, dost thou decoy, Thy golden sunshine to enjoy, Who hopes aye kind thou’lt be, From rival lovers free, Nor reckons aught of treacherous gale. Ah, luckless wights who spread their sail Of perils unaware Hid ’neath that surface fair! For me on Neptune’s temple wall A votive tablet’s lines recall My dripping garments brought To pay for rescue wrought.",
        "author": "Lionel Lancelot Shadwell",
        "date": 1920 },
{
        "text": "What dainty youth, Oh, Pyrrha fair, Makes love to you so ardently. For whom d’you bind your golden hair So gracefully. He gets to love you more and more, Yet when he finds that frequently You’re false to him, he’ll then deplore Your perfidy. And when he knows you only meant To flaunt and treat him scurvily, He’ll feel a wreck, and then resent Your treachery. And I, though often tempest tossed, Yet thank the Gods most fervently That through love’s turmoil, I have crossed Successfully.",
        "author": "William Frederick Lloyd",
        "date": 1920 },
{
        "text": "Tell me, my Pyrrha, in some cool alcove What slender scented stripling on thee presses Mid roses rife his ardent tale of love? For whom dost thou entwine those golden tresses Artlessly neat? Alas, how oft, deploring Woman’s inconstancy and fate unkind, Shall he survey the ocean tempest roaring And there a charm all unsuspected find, Who now enjoys thee, dross for gold mistaking, Thinking thy favour will for aye endure, Nor fears the fickle breeze’s rude awaking? Hapless are they who fall within thy lure Untried! For me,– the wall of yonder shrine, Where hangs my votive tablet, shall relate How that my raiment dank with dripping brine To bring Ocean’s god is consecrate.",
        "author": "Hubert Dynes Ellis",
        "date": 1920 },
{
        "text": "What gallant youth, my Pyrrha now, All perfumed o’er, caresses thee, In rosy dell? For whom braids thou Thy yellow hair so lovingly, With simple grace? Alas! the day When perjured vows he’ll oft bewail, That gods have changed, and him affray Will angry seas and blustering gale; Who, lulled beneath thy golden spell, Hopes thou will aye be free and kind, Nor other swain could love so well, But soon who’ll rue the change of wind! Ah, wretched whom thy gleam ensnares, Untried! I’ve hung in Neptune’s fane My dank robes, as the tablet bears, My debt to him who rules the main.",
        "author": "John Finlayson",
        "date": 1921 },
{
        "text": "What slender boy, fragrant with perfumed dews, On many a rose, thee, Pyrrha fondly sues, Under some pleasant cave? For whom confine thy wave Of amber tresses, in thy neatness plain? With tears, alas! how oft shall he complain Of adverse deities, And broken vows: and seas Rough with black blasts shall he, unused to ill, Wond’ring behold, who now, fond fool! his fill Takes, golden toy, of thee; Who thinks thee always free, And always kind, nor knows the fitful breeze. Hapless whoe’er untried thy beauties please! My life is all I boast, ’Scaped from that dangerous coast; The plank, last refuge from the stormy wave, Sinking, I vowed to Nereus– strong to save. Saved by his pow’r divine, I’ve fixed it on his shrine: There, too, yet dripping from the boisterous sea; Hung my wet garments, monuments of thee!",
        "author": "Anonymous, Morning Herald",
        "date": 1921 },
{
        "text": "What sleek young beau perfumed and smart Amid the roses plies the heart Of his embowered sweet? For whom within that pleasant grot Those golden tresses do you knot, So exquisitely neat? Today you’re gold for his delight And always calm and always bright; Ah! vain imaginings! No breath of doubt assails him now, Poor butterfly that sees the glow And has not singed its wings! How oft hereafter will he find The fickle gods have changed their mind And weep that faith is vain; And marvel as the storm-clouds sweep Black’ning across the ruffled deep, A landsman on the main! For me to consecrate the wall Where hangs a draggled overall A tablet stands, to mark My thanks to him whose powerful nod Secured– I mean the Ocean God My rescue from a shark.",
        "author": "Geoffrey Robley Sayer",
        "date": 1922 },
{
        "text": "Pyrrha, who is the perfumed youth who poses Beneath the pleasant grot, amid the roses, And courts you now; for whom you dress, with care And simple elegance, your golden hair? How oft will he, so credulous, be slighted By vows as lightly broke as lightly plighted? How oft deplore it that the Gods have changed, Withheld their favour, and become estranged? Shall he who fondly trusts you now be shocked By passion’s storms, supplanting smiles that mocked? As when the treacherous and uncertain breeze Swells to a gale and frets the summer seas. Oh wretched he, untried, who deems you fair! The sacred wall of Neptune’s fane shall bear A tablet, which shall my escape relate, And how my dripping garb I dedicate.",
        "author": "Edward Douglas Armour",
        "date": 1922 },
{
        "text": "Pyrrha, who is the dainty youth, With liquid perfumes reeking, Who now caresses you, forsooth, Your gracious favour seeking? For whom do you demurely wait Beneath the pleasant grot, With golden hair drawn smooth and straight, And coiled in simple knot? How often will you break the troth By you so lightly plighted? How often vent your senseless wrath So easily excited? As he who trusts the placid sea And gaily spreads his sails, Encounters unexpectedly Dark waves and furious gales, So, Pyrrha, he who trusts you fair, Inveigled by your glances, Poor wretch! will find he has to bear With shrewish whims and fancies. A votive tablet shall relate How I escaped the sea, And dripping garments dedicate, Oh Neptune, unto thee.",
        "author": "Edward Douglas Armour",
        "date": 1922 },
{
        "text": "Pyrrha, I sniff the air, and shrewdly guess That some well-perfumed scion of noblesse Is dallying with you among the roses, And winning your regard– as he supposes. While he disports himself within the grotto, I more than half suspect he’s trying not to Commit a breach of etiquette, and stare At the perfection of your golden hair. How he must marvel at the care and toil Expended to produce that simple coil! How wonder whether all that glorious sheen Is due to nature, or to brilliantine! Do you purpose to lead him such a dance That even Deities will look askance; To make him stand amazed that vows and oaths You break as often as you change your clothes? Oh Pyrrha, you remind me of the sea, Deceitful in its calm placidity; For he who trusts in you will never fail To find himself the sport of storm and gale. The toga which I wore when courting thee I dedicate to Neptune of the sea; And in his fane a tablet made of brass Shall show how I escaped a fickle lass.",
        "author": "Edward Douglas Armour",
        "date": 1922 },
{
        "text": "What scented youth now pays you court, Pyrrha, in shady rose-strewn spot Dallying in love’s sweet sport? For whom that innocent-seeming knot In which your golden strands you dress With all the art of artlessness? Deluded lad! How oft he’ll weep O’er changed gods! How oft, when dark The billows roughen on the deep, Storm-tossed he’ll see his wretched bark! Unused to Cupid’s quick mutations, In store for him what tribulations! But now his joy is all in you; He thinks your heart is purest gold; Expects you’ll always be love-true, And never, never, will grow cold. Poor mariner on summer seas, Untaught to fear the treacherous breeze! Ah, wretched whom your Siren call Deludes and brings to watery woes! For me– yon plaque on Neptune’s wall Shows I’ve endured the seaman’s throes. My drenched garments hang there, too: Henceforth I shun the enticing blue.",
        "author": "Grant Showerman",
        "date": 1922 },
{
        "text": "What perfumed gallant now pursues And hotly woos You, Pyrrha, where your roses bloom Or in your grotto’s grateful gloom? For whom do you now bind your hair, O blonde, O fair Those bright, gold braids, so smooth and neat, Deceitfully demure and sweet? Your love is now a tranquil sea, And on it, he A little hour serenely sails O changing gods! O mounting gales! Who thinks you’ll be forever kind Will change his mind, Will learn to wonder and to guess At fickle seas– and faithlessness. All who before him sailed that sea As trustingly, As he who, happy, trusts you yet, Are shipwrecked now; their garments, wet. One fate your lovers suffer, all. On Neptune’s wall My tablet and wet garments prove I, too, have sailed that sea of love.",
        "author": "Roselle Mercier Montgomery",
        "date": 1922 },
{
        "text": "Pyrrha! who is the youth whom you inspire, Where sheds the odorous rose its liquid fire? Who now in yon sequestered garden-plot, Pours in your ear his exigent desire? Do you for him your bright gold tresses bind? Can swathes so coiled forswear a gentle mind? Him, your feigned smile your glance demure have led, To wreck his happiness through folly blind. Your mien decorous oft shall he lament, Bewail changed deities and fortune spent. Regardless now, black winds he soon shall seek, Brave the wild surf and darkling firmament. His reason, thrall to your alluring smile, His hope, the hours more amiable beguile! His expectation reckons not the gale, Nor doubts his blissful heedlessness assail! Lost they, who blindly doting on your face. Prove at their cost your callous lack of grace! Ye gods! to whom I once wet garment hung, A mural slab now marks the sacred place!",
        "author": "Leonard Chalmers-Hunt",
        "date": 1925 },
{
        "text": "Slim, young, and essenced, Pyrrha, who On roses couched is courting you? Whom charms, in your sweet grot, The bright hair’s single knot, The choice plain dress? How oft he’ll cry False gods, false faith with tears, and eye, Poor novice, seas that change Storm-lashed to black and strange, Who now enjoys you, thinks you gold, Dreams you will love him still, still hold No hand but his, nor knows Winds change. Alas! for those Who trust your sheen. On temple-wall My votive tablet proves to all That Neptune earned his fee, Those dripping clothes, from me.",
        "author": "Hugh MacNaghten",
        "date": 1926 },
{
        "text": "Who is he, Pyrrha? Who the slender youth, The essenced lad, that now reposes A-courting thee amid the roses Within the pleasant grot? For whom, in sooth, Bindst thou thy golden hair, Artless yet trim? ... Ah me! how oft shall he Deplore thy faith, the gods deplore, Alike inconstant both, and sore Astonished view– new sight!– the raging sea By murky winds convulsed! He that thee holds the Queen of Hearts to-day, Poor innocent, who nothing knows Of shifting gales, whose hopes suppose Thou wilt be alway free, true love alway! Alas! alas! for men That know thee not but see thee dazzling fair! Me? On the temple’s sacred wall A votive tablet shows to all That I have hung my dripping clothes up there To the god that rules the sea.",
        "author": "Alexander Falconer Murison",
        "date": 1931 },
{
        "text": "In liquid odours steeped, what slender boy Courts thee mid roses nestling sweet and coy, My Pyrrha, in some pleasant grot? For whom thy golden hair dost knot, So neat, so simple? Ah! how oft shall he Troth changed, and gods, bewail, and watch the sea In wonder with black squalls disguised, Poor novice, disillusionized; Who holds, fond heart, thy love so golden dear, Nor dreams how soon the treacherous breeze can veer; Hopes thee for ever fancy-free, For ever kind. Ah! wretches be They whom, untried, thou dazzlest. As for me, A votive tablet shews for all to see, That I have hung in Neptune’s shrine, My garments drenched with brine.",
        "author": "Charles Reginald Haines",
        "date": 1933 },
{
        "text": "What lad with perfumed locks to-day Courts you amid the roses gay, Beneath some pleasant grotto’s shade? For whom, fair Pyrrha, do you braid, Simple and neat, your yellow hair? How oft full soon in wild despair On broken faith he will exclaim, And Heaven’s deluding favours blame; Perplexed and wondering will mark The rising gale and waters dark, Who now enjoys your golden mood, Trusting to find you kind and good For evermore and fancy-free. Poor simple youth! Unwitting he Of that tempestuous blast malign! Ill-fated those, for whom you shine Smooth and serene, as yet untried! For me, these sea-drenched clothes beside The votive painting on the wall My late escape from death recall: This offering due I make to thee, Great Neptune, ruler of the sea.",
        "author": "H. B. Mayor",
        "date": 1934 },
{
        "text": "What graceful youth, mid roses rich and rare, Whilst liquid scents perfume the ambient air, Now courts thee, Pyrrha, in thy grottoed lair? For whom dost thou now bind those auburn locks? Their simple neatness all their charm evokes. Ah me! How soon, in raging flood immersed, Will he, in tears, bemoan his fate accursed; His fond hopes blighted, and his life reversed. He who, at present, basks beneath thy smiles; Who thinks thee loving, knowing not thy wiles. Alas! for those whom thine allure beguiles. But Horace, now defended from the gale, Has hung Love’s dripping garment on the pale. A votive tablet in the Seagod’s fane Shows he will never launch forth to sea again.",
        "author": "Major Alfred Mailtland Addison",
        "date": 1935 },
{
        "text": "Pyrrha, what slender, scent-besprinkled youth now in your rose-decked bower with his whole heart worships you, while your flaxen locks you smooth with all the sweet simplicity of art? How often in the days to come shall he the perjured faith of fickle gods deplore, poor innocent, when o’er his placid sea wild billows lashed by sullen tempests roar! Trustful he revels in your golden beams, thinking you all unfettered, all his own, gentle and kind; nor for a moment dreams that cloudless skies with treacherous storms may frown. Unhappy victims, dazzled by your rays! Long since have I adorned with votive sign the sacred wall, and hung in grateful praise my dripping garments at the sea-god’s shrine.",
        "author": "Gilbert F. Cunningham",
        "date": 1935 },
{
        "text": "What slender boy with sleekly perfumed head Now courts you, Pyrrha, soft on rose leaves lying In some deep-shadowed cave? For whose sake do you smooth your auburn hair, So lovely in your cool simplicity? Alas, how often will he soon regret Your broken faith and his changed destiny, And tremble, unprepared, When first the dark winds rage across the sea, Stirring the rough-tipped waves to sudden foam. Now he enjoys your golden smiles, believing, In fatuous hope, that he alone is loved And you will always smile Not knowing yet how fickle is that breeze. Unhappy men, whom that bright sea allures! To them those untried waves seem smooth and clear, But as for me, my dripping garments hang Within the sea god’s shrine, And I have placed a votive tablet there That proves my gratitude to Neptune’s power.",
        "author": "Anne Thorburn Cleveland",
        "date": 1935 },
{
        "text": "What lovely lad bedewed with liquid scents Now courts you, Pyrrha, under pleasant cave ’Midst frequent flowers? In your excellence So natural, your golden locks you swathe Behind for whom? How oft his trusting mind And alienated gods will he bewail, And at the ocean, stirred by gloomy wind, Unused to such a pageant, will he quail. But now, too trustful, he enjoys the days Of love with you, and ignorant that the gale Is fickle, wants you calm and quiet always; For those on whom you shine untried are pale. The temple wall by votive tablet shows My dripping clothes to god of sea repose.",
        "author": "J. W., The Canberran",
        "date": 1935 },
{
        "text": "What youth as fresh as any flower, Pyrrha, is sighing in your bower: For whom is tied that yellow hair With careless care? How often shall he cry, alas! That faith and truth are frail as glass, And gasp when gales... no seaman he!... Convulse the sea. He dreams you golden to the end, Forever fancy-free, his friend, Nor knows what tricks the zephyrs play Most hapless they For whom as yet untried you shine! My dripping clothes still salt with brine (The votive tablet proves the vow) Are Neptune’s now.",
        "author": "Maurice Baring",
        "date": 1936 },
{
        "text": "What lissom lad, besprent with limpid perfumes, Mid roses– roses for your love pleads hard Pyrrha, ’neath some sweet grot? For whom Now loop you back your golden tresses So dainty-simple? Well-a-day! how oft Shall he wail troth forsworn, mourn Gods estranged, And stare with inexperienced wonder At waters rough with lowering storms Who finds you sweet now– duped!– his golden girl Who hopes that you will be aye untrespassed on, Will be a darling aye nor knows The breeze how faithless! Woe to them To whom you are a sunshine sea– untried! For me the temple-wall by votive picture Tells I have hung my weeds of shipwreck There to the God who rules the Sea.",
        "author": "Arthur Way",
        "date": 1936 },
{
        "text": "What graceful youth, whose bath with perfumes made, Now wooes you in your rosy-bowered shade, For whom, you, Pyrrha, braid your golden hair And, unadorned, your garb so coyly wear? Your broken faith, and gods besought in vain, Alas, will often wring his heart with pain, And when with rage your stormy mind is raised, Unused to such a scene, he’ll stand amazed; He, trusting, now enjoys your sunny smile And deems you ever fond and free from guile, To whom your fitful moods are still unknown Ah, wrecks! on whom your tempting eyes have shone! Behold within the Sea-god’s temple set My votive scroll and clothes all dripping wet, That will a grateful testimonial bear, Of life redeemed from wreck through Neptune’s care.",
        "author": "John B. Quinn",
        "date": 1936 },
{
        "text": "Who’s that slip of a youth, covered with roses and Bathed with liquid perfume, urging hid suit with you, Pyrrha, under yon pleasing bower? Why do you tie up your golden locks, Pretty enough as you are? Ah me, how oft will he Grieve that you’re without faith, and that his gods have changed! Does that innocent soul, who Now enjoys you and thinke you gold, Hope you’ll always be free: hope you will always be Lovely? Poor fellow, not knowing the tricky gale! How he’ll gasp when he sees the main Rough and stormy! Unhappy they who Don’t know you and still think you a gem. A votive Tablet, placed on the holy sanctuary well, proclaims I’ve devoted my wet clothes To the powerful god of the sea.",
        "author": "Justin Loomis Van Gundy",
        "date": 1936 },
{
        "text": "Pyrrha of the golden hair, Dressed with elegance and care, What youth now, with perfumed tresses, Is enjoying your caresses In some cave with roses strewn? Well I know that all too soon He will find his Pyrrha cold, All her kisses fairy gold. Drifting trustful, sails untrimmed, Soon he’ll feel a shifting wind. Pyrrha of the golden locks Drives his bark upon the rocks. Foolish, inexperienced youth, Thinks that you are his, forsooth! All are hapless whom your smiles Gladden ere they learn your wiles. Once I followed, fool as he. Out on Cupid’s untried sea: Shipwrecked too. Now on the wall Of Love’s fane, in sight of all, I have hung my garments dripping, Like a votive tablet, fitting Pledge that I’ll hereafter be Ware your siren witchery.",
        "author": "Herbert O. Williams",
        "date": 1936 },
{
        "text": "Who the young Adonis now, Wealth of roses on his brow, All besprent with attar-dew, Pleads his passion true Neath the grotto’s cool retreat? Ah! whose eye to greet, Braidest thou each golden tress, Simple in thy daintiness? Oft, alas, he’ll weeping learn Heaven’s smiles to frowns may turn, Troth be broken: oft will stare Wildered, unaware How black lowering storms can comb Waters into foam: While thy golden smiles entice, Fool in folly’s paradise. Hoping thou wilt ever be From all other fancy free; Hoping thou wilt ever prove Worthy of his love; Knowing naught of veering breeze. O’er uncharted seas Sail those hapless ones who be Dazzled by thy radiancy. Let the holy wall declare By the pictured tablet there Offered with thanksgiving vow This my story, how Once my raiment dripping wet, Pyrrha, there I set Hanging high, a rescue-fee To the mighty God of sea.",
        "author": "J. Lee Pulling",
        "date": 1939 },
{
        "text": "What lissom boy among the roses, Sprinkled with liquid scents, proposes To court you in your grotto, fair Pyrrha? For whom is your blond hair Bound, with plain art? Alas, how often Will he bid changed gods to soften, Till, poor landlubber, he finds The sea so rough with inky winds: Who now, poor gull, enjoys you gold And always careless, always bold To love, hopes on and never knows The gold is tinsel. Sad are those For whom you shine, untried. For me, Beholden to the great god of the sea A votive tablet will recall Drenched garments on his temple wall.",
        "author": "Keith Douglas",
        "date": 1939 },
{
        "text": "What graceful boy, suffused with liquid scents Makes love to you among the roses now? Who, Pyrrha, who, in a pleasant forest glade? For whom do you do your auburn hair In elegance so exquisitely plain? Today, perhaps, he mourns for Faithfulness And for the gods who have deserted him; And like a country lad who’s gone to sea For the first time in his life, his eyes are wide, Marvelling at oceans bitter with black winds. Too trusting in the golden hours is he Who now enjoys you; how he longs for you, Faithless and lovable one! Little knows he How fickle is the breeze. Unhappy those untried On whom you blow! And as for me Look on the temple wall where the tablets are, And you will see my damp clothes hanging there As thanks to the God who is Master of the Sea.",
        "author": "Quincy Bass",
        "date": 1940 },
{
        "text": "Pyrrha, what essenced youth with ardours bold Pursues thee now? for whom hast spread In thy delicious bower a roseleaf bed, And wrought mean thy lovely head That easy miracle of curling gold? Alas! how soon the hapless boy shall rue Thy broken faith, the kindly gods gone cold, And with amazement wake to view Black sudden winds lash up the seas Where now, unwitting of the treacherous breeze, He sails in blinded bliss, and holds thee true, Deeming thy sweet accommodating mood Set fair. Poor souls, I bode ye little good Who know not what those smiling deeps conceal. For me– my wreck is an old tale: Long since my votive scroll and garments wet, In Neptune’s temple hung, avowed my debt.",
        "author": "Edward Marsh",
        "date": 1941 },
{
        "text": "What slender stripling now Reclined on roses in some shady cave, With liquid scents imbued, dost thou enslave, Pyrrha, for whom dost thou Thy golden hair upbind, Bewitching in thy simple-seeming dress? How oft alas! shall he with tears confess The fickle gods unkind: And marvel to behold The storms and tempests of thy angry mood, Who thinks to find thee ever kind and good And deems thy heart pure gold. Ah! ’ware the shifting breeze, Lover untried! For me, my dripping weeds In Neptune’s fane are hung;– the tablet reads: I have escaped the seas.",
        "author": "John Seymour Blake-Reed",
        "date": 1941 },
{
        "text": "Pyrrha, what graceful boy now to your rosy Couch besprinkled with liquid odours Under your cave are you tying? For whom sprucely, but simply Curls that gold head? ... Ah, how many tear-drops For that too volatile heart of yours he’ll be shedding And for the changed Fates! Like as the waves unstable I admire astonished him who still credulous Takes joy in you as golden pure, and amiable, Hopes so to prove you ever, Of winds’ misfaith unwitting! But O what grief will blast him unawares! Already my dank garments with a votive Tablet upon the holy wall I’ve hung up to the ruler of the sea…",
        "author": "G. S. Fraser",
        "date": 1944 },
{
        "text": "What slim nard-scented youth, in rosy bower ’Neath some cool grot, Pyrrha with hair of gold, on thee doth shower His kisses hot? Sure, ’tis for him, those seeming-simple locks With art new-twined! Soon thee and fate both false, when in thy toils Poor wretch, he’ll find, As on a sea darkened with sudden gusts, Who art to-day All sun and smiles, all free, and so, he trusts, Will ever stay. Poor fools, who know thee not! My dripping gear Hung up of late In Neptune’s fane, in picture, shows how near I met my fate.",
        "author": "Frederick Charles William Hiley",
        "date": 1944 },
{
        "text": "What slender youth with perfumes all bedewed Amid the roses courts you, Pyrrha, now In pleasant grotto’s shade? For whom Do you arrange your hair In simple elegance? Alas, how much He will lament your changed faith, and fate, And marvel oft in great surprise At stormy seas of love, He who believes you now a golden lass And hopes, not minding favor’s fickle breeze, That always fancy-free you’ll be And always lovable. O wretched they for whom you shine, untried! For me, my votive gift on temple wall Is proof that I to Neptune have My dripping garments hung.",
        "author": "Ella H. Thompson",
        "date": 1944 },
{
        "text": " Pyrrha, what graceful youth among the roses, Sprinkled with liquid unguents, in a cave Loves thee? For whom thy yellow hair reposes In simple neatness? Ah, when he shall rave At the gods changing and thy promise broken, When he shall wonder newly at sea-storms Which the dark winds have raggedly awoken, Who now believes that he enjoys thy charms, Who sees thee, love on him alone bestowing Ever, who no deceptive breeze has known. Unhappy they on whom thou shinest unknowing. But, for myself, the temple wall has shown By votive tablet, and drenched clothes suspended To the sea’s god, my travelling is ended.",
        "author": "Lord Dunsany",
        "date": 1947 },
{
        "text": " What slender boy, with perfumes pure besprent, On couch of roses courts you, Pyrrha fair, In pleasant grotto pent? For whom your auburn hair, Bind you in artless neatness? Oft will he Bewail gods changed and good-faith gone awry, And, novice awed, the sea By lowering winds lashed eye, Who trustful finds you golden for the while, And hopes you’ll be as loving-fond for aye, Nor knows the breeze of guile. Ah, wretched men are they For whom you gleam, untried! A tablet slung On sacred shrine-wall does for me maintain That my wet weeds I’ve hung For Him who holds the main.",
        "author": "Skuli Johnson",
        "date": 1952 },
{
        "text": "What slender boy with perfumes rare In rose-decked grot doth woo thee now, For whom binds thou thy golden hair, O Pyrrha, on thy lovely brow, With sweet simplicity? Alack, How often shall he weep for woe, And curse his gods when seas are black And waves beat high and storm winds blow, And wonder how so sweet and fair And golden thou didst seem to be When loving, trusting, free from care, He dreamed of no deceit in thee. Ah, woe to those on whom the glow Of thy fair face hath cast its thrall Unknowing thy deceit I know. I’ve hung, O Neptune, on the wall, My dripping garments In thy shrine And votive tablet too to show How narrow an escape was mine!",
        "author": "Fred Bates Lund",
        "date": 1953 },
{
        "text": "What slender youth with perfume showered Woos thee in arches dim embowered? Pyrrha, for whom dost thou repair The admired disorder of thy hair? When nearer to life’s later bourn, How oft changed fortune shall he mourn! When at hard brow and mind’s rough gale His hopes of thee all-golden fail. Ill-starred for whom thou dost out-shine! He knows his thought, he knows not thine. Himself he deems thy life’s love-stay, No rival’s lure, no passion’s way. For me, that joy and strife of love Are past, doth that same tablet prove, Set in the wall of yonder shrine, A gift to the sea’s lord divine.",
        "author": "Augustus Ralli",
        "date": 1953 },
{
        "text": "What sleek and scented boy with roses flecked Importunes you within your pleasure-dome? For whom those golden tresses do you comb, Pyrrha so richly plain, demurely decked? Poor fool, he looks for love and constancy: How he will weep your Janus-smile, your trust! And mark, poor landsman, with what fearful wonder The blackening skies, the whirlwinds and the thunder! Poor innocent, he robs your treasury: How could he fancy that your gold would rust? Dupes whom you beacon over the uncharted foam! Here at the Storm-God’s shrine, I recollect, I paid my own thank-offering, lately wrecked; I hung my dripping clothes; I draggled home.",
        "author": "J. C. Saunders",
        "date": 1953 },
{
        "text": "Tell me, fair Pyrrha, in what bower of ease Deep strewn with roses you recline this hour; And who the scented youth you please and tease With lip and limb as fragrant as the flower? Whose the rapt gaze that falters as you tie That gold-spun hair in careless elegance, Snaring his heart with slow deceitful eye So that he lies your bondslave as in trance? Alas, how often shall the lad lament The faithless music of a breeze that blew To ravening gales of doubt and discontent, Who now finds all his gold of life in you. Poor dazzled fool, still thinking he’s the One! But I, delivered by the cleansing sea, Haste into Neptune’s temple from the sun To hang my dripping garments and be free.",
        "author": "William Keen Seymour",
        "date": 1953 },
{
        "text": "Who is the boy who still so fair In some cool cave beneath the ground Midst scents and roses strewn around Now wooes thee Pyrrha flower crowned For whom yet ignorant of care With simple art thou dost prepare To bind and braid thy golden hair. Poor boy, how bitterly he’ll weep Who too believing thinks to hold Thy faith, thy love as pure as gold For ever, never to grow cold. When wandering he shall wake from sleep To see dark clouds the waters sweep And whitening waves upon the deep. He thinks to find thee ever free And kind and gentle as the day Unconscious of the storms that may Lurk in soft airs, which round thee play. How they too rise relentlessly Poor fool, alas to look on thee But as an untried sparkling sea. The picture which in thankfulness When saved from deathly Neptune’s nod I vowed to that old watery God Now hangs aloft upon a rod. High in his temple with my dress Draggled and wet in my distress In token of your faithlessness.",
        "author": "J. K. Richmond",
        "date": 1953 },
{
        "text": "Who’s the slim youth that, where the roses spread, Perfumed with essences, woos thee to-day ’Neath the cool grotto, Pyrrha? For him thou braids thy golden hair Neat and demure. How oft will he bewail The loss of faith, the loss of Heaven’s favour And see with wild surprise The black storm ruffling all the calm. Now he enjoys thee and the golden prime And thinks, poor fool, his girl will ever be So fancy-free and kind, Not knowing how the breeze betrays, How the bright star proves false. I too have been Wrecked in those waves but lived to dedicate My votive, dripping garments To fickle Ocean’s mighty god.",
        "author": "N. C. Armitage",
        "date": 1953 },
{
        "text": " What slender lad, scented and sleek, Woos Pyrrha in her pleasant grot, Rose-bestrewn? For whom does she Her yellow tresses knot, So dainty plain? How oft shall he Of change of faith and gods complain, Wondering strangely at black winds That ruffle all his main? But now you’re golden, all his own Hope paints you as you’ve always been, Lovely, free– but breezes shift. Poor wretches, whom your sheen Untried allures! on temple wall The votive tablet tells of me Hanging up my dripping weeds To him who rules the sea.",
        "author": "Harold Mattingly",
        "date": 1953 },
{
        "text": "What comely youth, pomaded with the scent Of many a rose, entreats you, Pyrrha, now Within some welcome grotto? And for whom Do you, so gently graced, bind radiant hair? Alas! How oft his troth and fickle gods Shall he, untried, bemoan; and be amazed, Encount’ring rough seas and blackening storms, Who, trustful now, your golden self enjoys; And hopes, of changeful breezes unaware, That you will e’er be free and loving too. Unhappy they for whom you freshly shine! The temple by a votive tablet shows That for the Ocean God I have hung up My own wet garments to His Mightiness.",
        "author": "R. N. Green-Armytage",
        "date": 1953 },
{
        "text": "What pretty little boy With roses round his head, And perfume splashed all over him Insists upon your love for him, In the cool grotto’s shade? You bind your yellow hair With neat simplicity; But it’s your broken faith he’ll weep, His changed luck, the black winds that beat On the tormented sea, Poor wide-eyed innocent, Who thinks true gold is there, Hopes you’ll give all your time to him, And never be unkind to him; There’s treachery in the air! Those wretched novices That first see Pyrrha shine! A plaque on Neptune’s sacred wall Shows I’ve escaped, and hung up all My wet clothes at his shrine.",
        "author": "Adrian Collins",
        "date": 1953 },
{
        "text": "Who now, slender and young, deep in a charming bower Where wreathed roses abound, scents in his glistening hair, Begs thy favour? For whom dost Thou bright locks, Pyrrha, now array, All so daintily plain? Often, alas shall he Mourn how faith is estranged, gods are against him turned; Then he’ll face with amazement Storm-winds dark on a raging sea, Who now joys in the calm, trusting the golden mood, Sure thou’lt stay evermore kindly, and free for him, Nor sees how the deceiving Airs change. Sad is that the fate of those Who thy shining allure know not. A temple wall Bears my thanks to the Powers, pictured escape from wreck; There I’ve hung for an offering My wet clothes to the ocean god.",
        "author": "F. R. Dale",
        "date": 1953 },
{
        "text": "Pyrrha, what new and slender boy Scented with roses and with joy Is at you now in some green shade For whom your yellow hair you braid So sweet and simple? When he hears Vows broken, Gods unkind, what tears! Won’t they just take him by surprise Those rougher seas and blacker skies! Enjoying now your golden mood He trusts you always fond and good, And nothing knows of airs that change Unhappy boys: your charm is strange And dazzles them. I dared the gale And swam and lived to tell the tale. My dripping shirt hung up for me Shall thank the Ruler of the Sea.",
        "author": "John Mavrogordato",
        "date": 1953 },
{
        "text": "What slender youth amongst roses all abloom Comes courting thee, bedewed with perfumes rare O Pyrrha, in some pleasant cave? For whom Dost bind, so daintily, thy red-gold hair? Alas! how oft will he have cause to weep For changed gods, and faith and vows untrue, And wondering gaze, when sudden storm winds sweep With angry waves across the waters blue. Who fondly now enjoys thy golden hours Hoping thou still serene and kind wilt be As now, nor knows the fickle zephyrs’ powers The memory to beguile with flattery? Unhappy, sure, will be the fate of those On whom thou yet untried dost smile! For me The temple wall with dripping raiment shows My votive scroll to him who rules the sea!",
        "author": "E. H. Whishaw",
        "date": 1953 },
{
        "text": "What slender youth, bedewed with perfumes rare And crowned with roses, Pyrrha, courts thee now In some secluded cave? For whom dost thou, So daintily bind up thy auburn hair? Alas! how often shall he weep to find That vows are broken, altered gods unkind, And all unlooked-for, in surprised amaze On waters rough with sudden storm winds gaze, Who now enjoys thy golden hours with thee, And proudly hopes that ever tempest free And ever lovely thou wilt be, nor knows How fickle is the fairest wind that blows! Unhappy they, as yet unversed in guile, For whom with passing favour thou dost smile, On calm and sunlit days! What woe befell The tablet on the temple wall can tell, With dripping garments hung up piously In honour of the god who rules the sea.",
        "author": "E. H. Whishaw",
        "date": 1953 },
{
        "text": "What stripling slim, all-drenched in pungent perfumes, Who wooes you, Pyrrha, in some happy nook Festooned with clustering roses? For whom be those golden tresses. So trim so dainty? Ah how oft shall he bewail Reverse of fortune and plighted troth betrayed And wonder at smooth waters Billowing to winds and gales. He, who now is beaming beneath your sunny smiles, Who deems you ever loving, for ever loyal, Knows naught of changing breezes, Dear unsuspecting innocence! Ill-fated, whom your charms decoy an untried sea. A tablet set by me high on the temple wall Has vowed my sea-soaked garments Unto the god who rules the waves.",
        "author": "Natheus",
        "date": 1953 },
{
        "text": "Who, where clusters of rose guard, is the slender youth clasps you now with a sweet-scented embrace in cool grotto? Who’s it being wreathed for, this time, Pyrrha, the golden hair, chastely elegant?– One born to exclaim against faith and gods that have changed, fated to gaze upon waters billowing under darkening tempests in wild surprise, he who guilelessly now takes you for purest gold, thinks you’ll always be his, always adorable, still to learn how alluring breezes alter. Alas for those you, untrusted as yet, brightly illude! For me, yonder sanctified wall’s picturing tablet shows I survived to uphang there oozy garments to Ocean’s god.",
        "author": "James Blair Leishman",
        "date": 1956 },
{
        "text": "What slim elegant youth, drenched in effusive scent, now sits close to your side, Pyrrha, in some recess rich with many a rosebloom? Who loves smoothing your yellow hair, chic yet daintily plain? How many gods profaned, what indelible vows he will lament, and oh, what dark hurricane-lashed seas he will watch with a pallid cheek! Poor fool, golden he thinks you will for ever be, heart-free always, he hopes, always adorable yet knows not the deceitful offshore squalls. To a novice, you shine too temptingly bright. Here on the temple wall one small tablet of mine, offering up my clothes (all I saved from a shipwreck), says Thank God, that I just escaped.",
        "author": "Gilbert Highet",
        "date": 1957 },
{
        "text": "What lovely youth in what rose-scented lair Now lays his handsome head upon your lap? For whom now do you comb your yellow hair, And set with coy simplicity the trap? How oft will he deplore his wretched fate Like one who in fair weather sets to sea And strikes the tempest when it is too late To win again his lost tranquillity. Now he believes you golden through and through, Ever good-humoured, ever kind and sweet, He cannot find a single fault in you Nor tell true currency from counterfeit. Unhappy he who has not known your love, Unhappier he who has:– and as for me, That votive slab, these dripping garments prove I too have suffered shipwreck in that sea.",
        "author": "Alfred Duff Cooper",
        "date": 1959 },
{
        "text": "Pyrrha, who was that handsome youth With roses in his perfumed hair, Who in the grotto pledged his truth And claimed you as his only fair? You caught him with your golden curls No other gauds were needed then. He thought you were the best of girls, He’s now the most deceived of men. You were so equable and kind He never dreamed that storms could rise, Provoked by fickle change of wind To blacken all the summer skies. Unhappy those who have been caught By thy glad eye and winsome mien, Experience is so dearly bought And girls are not all what they seem. I too was near engulfed; my vest Hangs sodden in the sea god’s shrine As votive offering to attest How lucky this escape of mine.",
        "author": "John Allsebrook Simon",
        "date": 1959 },
{
        "text": "What slender Sirrah is after you now My Pyrrha? In cool shady bowers Who woos you today with his scent and his flowers? For whom are you binding your golden tresses? Who thinks, poor boob, your caresses Eternal, you innocent minx? Sure! The heavens smile fair on the smooth sea-face! In his arms you swear That your golden charms are his for keeps! But tomorrow changed are the skies, black and harsh there leaps The treacherous storm-driven billow. And he, poor fellow, will find Your mood has veered with the wind. Pity the innocent youth, who knows not the truth That experience buys of your lies And your smiles that beguile. For me, I have had it, my poppet! Never again! Half drowned but sane, Rescued when almost sunk, I have hung my dripping junk Upon his temple wall, who heard my call: To show my thanks and my vow To him, the Great God of the sea: This the last voyage for me.",
        "author": "Christopher Storrs",
        "date": 1959 },
{
        "text": "So who is this wears roses And all the scents of May? And what is the road, my lord, my pretty, You take this day? The road I take, old poet, Will lead to a bed of down; For my lady is waiting with tresses of fire And a plain silk gown. As yet thy love is a summer’s sea, And thy ship rides easy of keel, But when gods turn, then winds are arisen, Winds black as steel. As yet thy love is calm and kind, Thy love is the purest gold: But a wind is stirring, my lord, my pretty, A wind false and cold. For I knew her spring and her winter too, And scarce escaped from the brine: I bought hose of worsted and hung my silk In Neptune’s shrine.",
        "author": "Simon Raven",
        "date": 1959 },
{
        "text": "Pyrrha, what slender boy sprinkled with soft perfumes, within some pleasant cave courts you amid rose blooms? For whom are you binding back your hair of silken gold, graceful though unadorned? Poor lover! Times untold he will lament your light capricious loyalties, staring in pained surprise at the dark storm-tossed seas, who now enjoys the bliss your golden charms afford, pictures you ever free, ever to be adored, blind, in his trusting love, to the wind’s treachery. Woe to those innocents you dazzle! As for me, a tablet on Neptune’s wall declares that, safe ashore, I offered up to him the dripping clothes I wore.",
        "author": "Niall Rudd",
        "date": 1959 },
{
        "text": "What scented stripling do you walk with now? And in the dell, among the roses fair, Whose arms enfold you, Pyrrha: and for whom Bind you your yellow hair With simple grace? He will, alas, too soon Bewail your broken troth, and heaven’s caprice That sent him, all unknowing, to embark On such unfriendly seas. Now in his eyes you still are purest gold, And in his dreams you walk faithful and kind. The seas are calm, the heavens unclouded still, And gently blows the wind. Unhappy men who know not yet your worth. But I, engulfed, against uncounted odds Survived the storm, and treading the dry earth Have hung my seawet garments to the gods.",
        "author": "Alan McNicoll",
        "date": 1959 },
{
        "text": "What slim and sweetly scented boy presses you to the roses, Pyrrha, in your favorite grotto? For whom is your blond hair styled, deceptively simple? Ah, how often he’ll sob over your faithless conversions, staring stupidly at the black winds and wild seas. Hejhas you now, for him you have a golden glow, ever contented, ever loving he hopes, unaware of the tricky breeze. Poor things, for whom you glitter before you’re tried. The temple wall with its plaque serves notice: I have hung my wet clothes up and bowed to the sea god’s power. ",
        "author": "Joseph P. Clancy",
        "date": 1960 },
{
        "text": "Ah, what delicate lad sprinkled with liquid scents Now on many a rose pleads with you, Pyrrha, there In some favorite grotto? Now for whom do you bind, so trim, Your bright red-golden hair? How oft, alas! will he Weep your changed faith, changed gods! Marveling, he will gaze At seas roughened by black winds, Staring, stunned by the strange, new sight, Who, believing you gold, feels now but joy in you, Who now hopes you’ll be sweet, pleasant, and fancy-free; He knows naught of your false breeze, Shifting, treacherous! Wretched, they For whom, untried, you shine. Me– well, the sacred wall In a votive scene shows, plain on my tablet, that To the sea’s mighty god, long Since I hung up my dripping clothes.",
        "author": "Helen Rowe Henze",
        "date": 1961 },
{
        "text": "What perfumed youth pursues thee now Amid the lovely roses’ scent? O Pyrrha, what protested vow Does he in budding groves invent, For whom your golden coiffure’s tousled– simply elegant–? Too often will the lad lament Frail faith and alternating vow; And, unaccustomed to repent, He will repent his eager prow That swam your sea, turned vehement With black and bitter breezes now; Who now takes gullible delight In thinking you the purest gold, With him by day, alone by night, Available, yet single-souled, Who cannot see you for the light Of love, the fool’s gold. Unhappy they for whom you hold The charm of unexplored sight. My votive plaque, in pew enscrolled, Shows hanging on the trident bright My hat and coat, still dank and cold, To drip my praise of Neptune’s might.",
        "author": "John Crossett",
        "date": 1963 },
{
        "text": "What lightfoot lad pursues you now With sweet bouquets of roses? O Pyrrha, what protested vow Is that which he encloses? For when you picnic in the wood, Your golden hair tied in a snood, Your summer frock– O tell me how Conceals what it exposes. Poor boy, how soon from every tear He’ll take perpetual notion Of faith that’s false, of vows that veer On Love’s Pacific Ocean; And those black squalls of cruelty Which wrinkle on your silken sea Will leave him, who has yet to hear Them, sick from their cross-motion. He takes incredible delight In credulous reflections That you are ready, day and night, To share your blonde affections. Observe how hopefully he smiles, So unaware your golden wiles Can put in such beguiling light Your various imperfections. I pity them who follow you And dare attempt temptation, In ignorance that they pursue A bright hallucination. There is a chapel by the sea, Within a shrine, paid for by me, Where often, from my votive pew, I murmur my oblation.",
        "author": "John Crossett",
        "date": 1963 },
{
        "text": "What graceful lad with lavish crown Of rose, and perfumes dripping down Beneath thy grotto’s loveliness With passion, Pyrrha, doth thee press? For whom thy golden hair dost braid, Thine elegance so simply made? Alas! How oft will he bewail Thy fickle faith and gods that fail And stare aghast, untried in these, At thy black gusts and stormy seas, Whose now thy golden favour is, Who hopes thee dear and wholly his, For ever! Fool! He cannot know What cheating winds from Pyrrha blow. For whom thou art uncharted sea, Thou hast a glamour, wretched he! My seascape hung near temple door Shows, after shipwreck now on shore. That my damp rags I thankful gave To Neptune, monarch of the wave.",
        "author": "Frederick William Wallace",
        "date": 1964 },
{
        "text": "Bedded on roses, in a cave of pleasure What smooth boy, hair wet with sweet odors, Plies you with his need, Pyrrha? Tie back the loose gold treasure of your hair, Simple in your riches. Soon he will cry The changing gods and your inconstancy Like a new sailor finding out the sea’s Incontinence, and the black winds’ embrace Who thinks that all your gold is his, the pure Maiden’s gift of her first hopeful ardor. Fools mindless of the wind Try the trick currents where you beckon them, New fruit untasted on a distant shore. You can read on the temple wall the oath I swore When I nailed up, years ago, My dripping sailor suit on that locked door.",
        "author": "Edwin Watkins",
        "date": 1967 },
{
        "text": "What slim youngster, his hair dripping with fragrant oil, Makes hot love to you now, Pyrrha, ensconced in a Snug cave curtained with roses? Who lays claim to that casually Chic blonde hair in a braid? Soon he'll be scolding the Gods, whose promise, like yours, failed him, and gaping at Black winds making his ocean’s Fair face unrecognisable. He’s still credulous, though, hugging the prize he thinks Pure gold, shining and fond, his for eternity. Ah, poor fool, but the breeze plays Tricks. Doomed, all who would venture to Sail that glittering sea. Fixed to the temple wall, My plaque tells of an old sailor who foundered and, Half-drowned, hung up his clothes to Neptune, lord of the element.",
        "author": "James Michie",
        "date": 1973 },
{
        "text": " What slim youth, Pyrrha, drenched in perfumed oils, Lying in an easy grotto among roses, roses, Now woos, and watches you Gathering back your golden hair, With artless elegance? How many a time Will he cry out, seeing all changed, the gods, your promise, And stare in wondering shock At winds gone wild on blackening seas! Now fondling you, his hope, his perfect gold, He leans on love’s inviolable constancy, not dreaming How false the breeze can blow. Ah, pity all those who have not found Your glossy sweetness out! My shipwreck’s tale Hangs, told in colors, on Neptune’s temple wall, a votive Plaque, with salvaged clothes Still damp, vowed to the sea’s rough lord.",
        "author": "Cedric Whitman",
        "date": 1980 },
{
        "text": "What slender boy besprinkled with fragrant oils now crowds you, Pyrrha, amid the roses in some convenient grotto? For whom do you dress that yellow hair, so simply neat? Alas, how often he will weep at your and the Gods’ vacillations – oh he will be flabbergasted by rough seas and black gales, who now enjoys the illusion your worth is golden, who supposes you will be always available, always amiable, not knowing the breeze deceives. I pity those for whom you blandly glitter. A votive plaque on the temple wall",
        "author": "W. Guy Shepherd",
        "date": 1983 },
{
        "text": "What perfumed debonair youth is it, among The blossoming roses, urging himself upon you In the summer grotto? For whom have you arranged Your shining hair so elegantly and simply? How often will he weep because of betrayal, And weep because of the fickleness of the gods. Wondering at the way the darkening wind Suddenly disturbs the calm waters. Now he delights in thinking how lovely you are. Vacant of storm as the fragrant air in the garden Not knowing at all how quickly the wind can change. Hapless are they enamored of that beauty I Which is untested yet. And as for me? The votive tablet on the temple wall Is witness that in tribute to the god have hung up my sea-soaked garment there.",
        "author": "David Ferry",
        "date": 1998 },
{
        "text": " What graceful youth, bedewed with perfume, is embracing you, O Pyrrha, on beds of roses in the pleasant grotto? For whom have you braided with measured elegance your golden hair? Alas! How often shall he bewail your infidelity? the gods’ adversity? the unexpected black winds, the agitated seas, gazing aghast at that sudden change of weather? who now, all credulous, enjoys your golden altogether, hoping you ever available, forever lovable, ignorant of the treachery breathing now beside him O wretched are they to whom, untried, you seem all pure: a bride. Upon the sacred votive wall, see suspended now my dripping robes offered up in grateful devotion to the god that rules the ocean.",
        "author": "Sidney Alexander",
        "date": 1999 },
{
        "text": "What slender boy, Pyrrha, drowned in liquid perfume,urges you on, there, among showers of roses, deep down in some pleasant cave? For whom did you tie up your hair, with simple elegance? How often he’ll cry at the changes of faith and of gods, ah, he’ll wonder, surprised by roughening water, surprised by the darkening storms, who enjoys you now and believes you’re golden, who thinks you’ll always be single and lovely, ignoring the treacherous breeze. Wretched are those you dazzle while still untried. As for me the votive tablet that hangs on the temple wall reveals, suspended, my dripping clothes, for the god, who holds power over the sea.",
        "author": "A. S. Kline",
        "date": 2003 },
{
        "text": "What slender boy bathed in a flowing smell Courts you, Pyrrha, on roses Within some pleasant cave? Whom do you braid that golden hair for, Simple and neat? Ah, how often He’ll weep at how faith and gods change, And he’ll marvel, unaccustomed, At this rough sea that’s blackened by the wind. Credulous, he enjoys you now, golden one. Hoping you’ll be always free, always beautiful, He’s unaware of the changing wind. Unfortunate are those whom you, Untried, dazzle. The votive plank On the temple wall shows how I escaped: I’ve hung up my wet clothing In honor of the god of the sea.",
        "author": "David Bowles",
        "date": 2003 },

{
        "text": "What slight young man awash with fragrant scents pursues you, Pyrrha,1 on a rosy bed within a charming cave? For whom is your blond hair tied back in simple elegance? Alas, how often broken faith and fickle gods he’ll mourn and be amazed at seas turned hostile by black squalls! The innocent, he now enjoys you, thinks you are like gold, expects you always lovely and avail- able, unaware of treacherous gusts. Wretched those for whom you gleam untried: I too, as the votive tablet on the temple wall makes known, hung up to Neptune2 clothing dripping from the storm.",
        "author": "Jeffrey H. Kaimowitz",
        "date": 2008 },
{
        "text": "Who’s the slim boy pressing on you Among the rose petals, Pyrrha, Soaked with perfumes his body through In a secluded cave somewhere? Now you tie back your flaxen hair, Simple and neat beneath his gaze; Ah, but in tears he soon will swear Faith and the gods have changed their ways. He will stare out and watch the sea Boil at black winds. How raw he is, Who now enjoys you credulously, Hoping your golden self is his, Free to be loved, single, untied Always; not knowing the false breeze! I pity those who have not tried Your shining waters. Through with the seas, My plaque on Neptune’s temple wall Shows that in dedication I Have hung my sodden garments all In honour of his potency!",
        "author": "Stuart Lyons",
        "date": 2007 },

{
        "text": "What slip of a boy, all slick with what perfumes, is pressing on you now, o Pyrrha, in your lapping crannies, in your rosy rooms? Who’s caught up in your net today, your coil of elegant coiffure? He’ll call himself a sucker soon enough, and often, and rail at the breakers—God of his word, you of your faith. The darkest sort of thought will fill his form when breezes bristle, mirrors roughen—just you wait! So far, his seas are barely stirred. You are forever fair to his fairweather mind, and golden to his gullibility: no storms are forecast there, and no distress. What blind and wretched men— you’ve barely touched them, yet they find you gripping! Whereas I have tendered my last and best regards to the Gods of the wave, as temple tablets will attest. I’ve thrown off the habit, and hung up my wet suit there. (You see? It’s dripping.)",
        "author": "Heather McHugh",
        "date": 2014 },
{
        "text": "So who’s that pretty boy, soaked in cologne, grinding against you in the rose bushes near that pleasant grotto, Pyrrha? Is it for him that you do up your blonde hair, stylishly simple? Ah, how often he will be in anguish over fickle faith and fate, and be caught off guard – astounded – as if at the sea abruptly churned up by a dark gale. He may be enjoying you now – your radiance –always believing in your easy-going love, unaware of the deceptive way the wind blows.Miserable are they who’ve never basked in your glow. As for me – see my dripping clothes hanging on the holy temple wall as an offering for the powerful god of the sea? Well, they show that I’ve survived that particular storm.",
        "author": "Anonymous, arterialtrees.home.blog",
        "date": 2018 },
{
        "text": "What simple boy, having doused himself in perfume,hems you in on a bed of roses under cover of a pleasant cave? For whom do you, Pyrrha, simple in your elegance, arrange your golden locks? Ah, how many times will that boy cry over fickle faith and fickle fortunes and, in his insolence, will stand aghast at the oceans made rough by black storms; That trusting boy, who now enjoys you in all your magnificence and who always hopes you are available and always hopes you are loveable, is ignorant of your false charms. Wretched are those to whom you appear glamorous without knowing your true self. A sacred wall shows that I have suspended my wet clothes there as a votive prayer for the powerful god of the sea.",
        "author": "Melissa Beck",
        "date": 2018 }



]


# This sets up what we'll do when we clean the texts, removing punctuation and gives us the raw language, for better comparison across time periods
def preprocess(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

# Extracts texts, authors, and dates, which are organised in the translations.
texts = [t["text"] for t in translations]
authors = [t.get("author", "Unknown") for t in translations]
dates = [t.get("date", "Unknown") for t in translations]

# This actually does the preprocessing described above.
texts_clean = [preprocess(t) for t in texts]

# To more easily check results against a database, we'll add the first three words of each translation to the labels.
def get_first_three_words(text):
    return ' '.join(text.split()[:3]) if text else "[No Text]"

labels = [
    f"Version {i+1} ({authors[i]}, {dates[i]}): {get_first_three_words(texts[i])}"
    for i in range(len(translations))
]

# This makes the n-grams into vector scores, so they can be compared to one another.
#I first spent a long time working with a process called sentence-transformers, which is a more sophisticated natural language processor than an ngram approach, taking into account semiotic closeness, synonyms etc. - it was rewarding to some extent, but the margins for similarity were very small, because, of course, definitionally each translation of 'ad pyrrham' has the same semiotic weight, give or take some details, because they're all saying the same thing. The ngram approach is more revealing, though less sophisticated from a computing standpoint.
vectorizer = CountVectorizer(ngram_range=(2, 3)).fit(texts_clean)
ngram_matrix = vectorizer.transform(texts_clean)
similarity_matrix = cosine_similarity(ngram_matrix)

# This section will print a list of all the most common phrases.
# You can change the numbers after ngram range to look for 2 or 4 word phrases.
vectorizer_3gram = CountVectorizer(ngram_range=(3, 3)).fit(texts_clean)
threegram_matrix = vectorizer_3gram.transform(texts_clean)
threegram_counts = threegram_matrix.sum(axis=0).A1
threegram_phrases = vectorizer_3gram.get_feature_names_out()
top_3gram_indices = np.argsort(threegram_counts)[::-1][:20]

print("\nMost common four-word phrases:")
for idx in top_3gram_indices:
    print(f"{threegram_phrases[idx]} (count: {threegram_counts[idx]})")

# This produces a graph where each of the connected poems is a node, and there are lines showing strong connections between them.
# This is from an earlier version, with fewer translations in the dataset - it is less useful now, because there are so many connected texts that the image is quite hard to read. ---
# If you put in fewer translations above, you would get more readable data here, but it is still to some extent generating useful visual information about clusters.
G = nx.Graph()

# Sets the threshold for how strong a connection has to be to be shown. Anything below 0.001 won't show anything --- this is a problem.
top_percent = 0.005
sorted_similarities = np.sort(similarity_matrix.flatten())[::-1]
threshold_index = int(len(sorted_similarities) * top_percent)
threshold = sorted_similarities[threshold_index]

# Connections can be colour coded based on their strength, their 'weight'.
max_weight = np.max(similarity_matrix)
min_weight = np.min(similarity_matrix[similarity_matrix > threshold])

# Add edges for similarity above threshold
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = similarity_matrix[i, j]
        if sim > threshold:
            G.add_edge(i, j, weight=sim)

# This cuts out any poems that only show one strong connection, or none, to reduce noise in the image.
G = G.subgraph({n for n in G.nodes if G.degree(n) > 1}).copy()

# This draws the background of the graph
fig, ax = plt.subplots(figsize=(12, 10))
pos = nx.kamada_kawai_layout(G)

# This makes nodes bigger and smaller based on the number of connections.
degrees = dict(G.degree())
node_sizes = [degrees[n] * 150 for n in G.nodes()]  # Scale factor (adjust as needed)

# This decides which colour each line should be, and draws them.
for i, j in G.edges():
    weight = similarity_matrix[i, j]
    edge_color = (weight - min_weight) / (max_weight - min_weight + 1e-5)
    nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], edge_color=plt.cm.coolwarm(edge_color), width=2)

# This draws the nodes.
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, alpha=0.9)

# This draws the labels.
for i, (x, y) in pos.items():
    ha = 'left' if x < 0 else 'right'
    plt.text(x, y, labels[i], fontsize=8, ha=ha, bbox=dict(facecolor='white', alpha=0.7))

plt.title("Translation Similarity Network (Top 0.5%)")
plt.axis('off')
plt.show()


# This ranks the poems by similarity, to show influence given and received. It takes no account of date, so influence given shows up the same way as influence received.
influence_scores = similarity_matrix.sum(axis=1)
sorted_indices = np.argsort(influence_scores)[::-1]

print("\nVersions ranked by similarity to others:")
for idx in sorted_indices:
    similar_versions = [
        labels[j] for j in np.argsort(similarity_matrix[idx])[::-1]
        if similarity_matrix[idx, j] > threshold and j != idx
    ]
    print(f"{labels[idx]}: similar to {', '.join(similar_versions)}")

from collections import defaultdict

# Converts dates to numbers, and rounds to nearest multiple of fifty.
def get_half_century(date_str):
    try:
        year = int(date_str)
        return (year // 50) * 50  # e.g., 1887 -> 1850, 1923 -> 1900
    except:
        return None

# Each poem is given a fifty year group.
half_centuries = [get_half_century(d) for d in dates]

# This organises the groups by fifty year intervals
half_century_groups = defaultdict(list)
for idx, period in enumerate(half_centuries):
    if period is not None:
        half_century_groups[period].append(idx)

# This sorts periods
sorted_periods = sorted(half_century_groups.keys())

# This generates an average score for each period
average_scores_by_half_century = []
for period in sorted_periods:
    indices = half_century_groups[period]
    if indices:
        avg_score = np.mean(influence_scores[indices])
        average_scores_by_half_century.append((period, avg_score))

# This part plots the results
plt.figure(figsize=(12, 6))
x_vals = [p for p, _ in average_scores_by_half_century]
y_vals = [s for _, s in average_scores_by_half_century]

plt.plot(x_vals, y_vals, marker='o')
plt.xlabel("50-Year Period")
plt.ylabel("Average Similarity to Other Versions")
plt.title("Average Translation Similarity by 50-Year Period")
plt.xticks(x_vals, [f"{p}s" for p in x_vals])  # prettier x-axis labels
plt.grid(True)
plt.show()

# --- Bar chart: Number of translations per 50-year period ---
period_counts = {period: len(indices) for period, indices in half_century_groups.items()}
sorted_periods = sorted(period_counts.keys())
counts = [period_counts[p] for p in sorted_periods]

plt.figure(figsize=(12, 6))
plt.bar([f"{p}s" for p in sorted_periods], counts, color='teal', alpha=0.7)
plt.xlabel("50-Year Period")
plt.ylabel("Number of Translations")
plt.title("Number of Translations per 50-Year Period")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# --- Compare a new poem ---
new_poem = """
What Youth in rosy Bower laid, His Locks with liquid Odours spread, Now hugs thee to his panting Breast? And thinks no Mortal half so blest? For whom dost thou, enchanting Fair, In Ringlets wreath thy flowing Hair? For whom, my Pyrrha, dost thou deign To deck thus elegantly plain? The unwary Wretch, who sees no Guile, Drinks Poison in at every Smile, And figures to his flattering Mind Thee, always vacant, always kind; Unwont to see, unwont to hear One chiding Word, or Look severe; How shall he view, with secret Dread, That heavenly Face with Clouds overspread? How often curse his fatal Love? His gods? who so inconstant prove. Ah, hapless they! who view that Face Adorned with every winning Grace; Unknowing Pyrrhas fickle Heart Full fraught with all-deceiving Art. In yonder votive Tablets read How I, from dreadful Ship-wreck freed, My dropping Weeds hung up to Thee, Great Neptune, Ruler of the Sea."""
new_poem_clean = preprocess(new_poem)
new_vector = vectorizer.transform([new_poem_clean])
new_similarities = cosine_similarity(new_vector, ngram_matrix).flatten()

print("\nSimilarity of NEW poem to each translation:")
for idx in np.argsort(new_similarities)[::-1]:
    print(f"{labels[idx]}: {new_similarities[idx]:.4f}")

average_similarity = np.mean(new_similarities)
print(f"\nAverage similarity of new poem to all translations: {average_similarity:.4f}")

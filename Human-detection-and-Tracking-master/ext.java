package imagetransforms;

import java.awt.Color;

import sedgewick.Picture;

public class Transforms {

/**
* This one is solved for you.
* @param source
* @param target
*/

public static void flipHoriz(Picture source, Picture target) {

for (int x = 0; x < source.width(); x++) {

for (int y = 0; y < source.height(); y++) {

//

// Convince yourself that otherX is the x coordinate flipped,

// on the other side of the image by doing the following:

// Check that this is true when x == 0

// then otherX = source.width() - 1

// which is indeed the rightmost pixel

// Check that this is true when x == source.width()-1

// then otherX = 0

// which is indeed the leftmost pixel

//

int otherX = source.width() - 1 - x; // otherX is the mirror of x

Color c = source.get(otherX, y); // get the Color at the mirror point of the source

target.set(x, y, c); // and set it at x,y in the target

}

}

}

public static void flipVert(Picture source, Picture target) {

for (int x = 0; x < source.width(); x++) {
for (int y = 0; y < source.height(); y++) {

int otherY = source.height() - 1 - y; // otherY is the mirror of y

Color c = source.get(x, otherY); // get the Color at the mirror point of the source

target.set(x, y, c); // and set it at x,y in the target

}
}
}

public static void flipHorizLeftHalf(Picture source, Picture target) {
for (int x = 0; x < source.width()/2; x++) {
for (int y = 0; y < source.height(); y++) {
Color c = source.get(x, y);
target.set(x, y, c); 
}
}

for (int x = source.width()/2; x < source.width(); x++){
	for(int y=0 ; y <source.height(); y++){
		int otherX = source.width()-x-1; // the -1 here might cause problem, try deleting it if something is wrong
		Color c = source.get(otherX, y);
		target.set(x,y,c);
}
}
}

public static void flipVertBotHalf(Picture source, Picture target) {
for (int x = 0; x < source.width(); x++) {
for (int y = 0; y < source.height()/2; y++) {
Color c = source.get(x, y);
target.set(x, y, c); 
}
}

for (int x = 0; x < source.width(); x++){
	for(int y= source.height()/2 ; y <source.height(); y++){
		int otherY = source.height()-y-1; // the -1 here might cause problem, try deleting it if something is wrong
		Color c = source.get(x, otherY);
		target.set(x,y,c);
}
}


}

public static void gradient(Picture target) {
for (int x = 0; x<target.width(); x++){
	for(int y =0; y<target.height(); y++){
		int amountRed = (int)(255*x/target.width());
		int amountGreen = (int)(255*y/target.width());
		target.set(x,y, new Color(amountRed, amountGreen, 128));
	}
}
}

public static void edgeDetect(Picture source, Picture target) {

// FIXME

}

public static void digitalFilter(Picture source, Picture target) {

// FIXME

}

}

package RockPaperScissors;

import java.util.Random;

import cse131.ArgsProcessor;

public class RPS {

public static void main(String[] args) {

// TODO Auto-generated method stub

ArgsProcessor ap = new ArgsProcessor(args);

String [] player1 = {"rock","paper","scissor"};
String [] player2 = {"rock","paper","scissor"};
int rounds = ap.nextInt("How many rounds do you want to play?");
int p1wins = 0;
int p2wins = 0;


for (int i =0; i<rounds;i++){
  int p2 = (int)(Math.random()*3);	// p2 stores the int of player2's move: random
  int p1 = (i%3);					// p1 stores the int of player1's move: rotate
  									// 0:rock, 1:paper, 2:scissor
  System.out.println("Round"+i+": Player 1:"+player1[p1]);
  System.out.println("Round"+i+": Player 2:"+player2[p2]);
  if (p2 == 0){
  	if (p1 ==0) {System.out.println("Round"+i+": Game ties");}
  	if (p1 ==1) {System.out.println("Round"+i+": Player 1 wins"); ++p1wins;}
  	if (p1 ==2) {System.out.println("Round"+i+": Player 2 wins"); ++p2wins;}
  }

  if (p2 == 1){
  	if (p1 ==0) {System.out.println("Round"+i+": Player 2 wins"); ++p2wins;}
  	if (p1 ==1) {System.out.println("Round"+i+": Game ties");}
  	if (p1 ==2) {System.out.println("Round"+i+": Player 1 wins"); ++p1wins;}
  }

  if (p2 == 2){
  	if (p1 ==0) {System.out.println("Round"+i+": Player 1 wins"); ++p1wins;}
	if (p1 ==1) {System.out.println("Round"+i+": Player 2 wins"); ++p2wins;}
  	if (p1 ==2) {System.out.println("Round"+i+": Game ties");}
  }
 }
 System.out.println("Total p1wins: "+p1wins+"out of "+rounds+" rounds");
 System.out.println("Total p2wins: "+p2wins+"out of "+rounds+" rounds");
}